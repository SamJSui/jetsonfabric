package agent

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

type Server struct {
	backend runtimeclient.ChatBackend
	nodeID  string
}

type Option func(*Server)

func WithNodeID(nodeID string) Option {
	return func(s *Server) {
		s.nodeID = nodeID
	}
}

func NewServer(backend runtimeclient.ChatBackend, opts ...Option) *Server {
	server := &Server{backend: backend}
	for _, opt := range opts {
		opt(server)
	}
	return server
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	mux.HandleFunc(api.RouteLayerSplitStage, s.handleLayerSplitStage)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "jetsonfabric-agent"})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if s.backend == nil {
		writeError(w, http.StatusServiceUnavailable, errorRuntimeUnavailable, "runtime backend is not configured")
		return
	}
	defer r.Body.Close()

	var req chat.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "messages must contain at least one message")
		return
	}

	resp, _, err := s.backend.Complete(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, errorRuntimeRequestFailed, err.Error())
		return
	}
	resp.Route = nil
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleLayerSplitStage(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var req layersplit.ActivationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	if req.SessionID == "" {
		writeError(w, http.StatusBadRequest, errorMissingSessionID, "session_id is required")
		return
	}
	if req.RequestID == "" {
		writeError(w, http.StatusBadRequest, errorMissingRequestID, "request_id is required")
		return
	}
	if req.ModelID == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model_id is required")
		return
	}
	if req.LayerEnd <= req.LayerStart {
		writeError(w, http.StatusBadRequest, errorInvalidLayerRange, "layer_end must be greater than layer_start")
		return
	}
	if req.Payload == "" {
		writeError(w, http.StatusBadRequest, errorMissingPayload, "payload is required")
		return
	}

	start := time.Now()
	nodeID := s.nodeID
	if nodeID == "" {
		nodeID = req.NodeID
	}
	resp := layersplit.SyntheticStageResponse(req, nodeID, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

type errorCode string

const (
	errorInvalidJSON          errorCode = "invalid_json"
	errorInvalidLayerRange    errorCode = "invalid_layer_range"
	errorMissingModel         errorCode = "missing_model"
	errorMissingMessages      errorCode = "missing_messages"
	errorMissingPayload       errorCode = "missing_payload"
	errorMissingRequestID     errorCode = "missing_request_id"
	errorMissingSessionID     errorCode = "missing_session_id"
	errorRuntimeUnavailable   errorCode = "runtime_unavailable"
	errorRuntimeRequestFailed errorCode = "runtime_request_failed"
)

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, code errorCode, message string) {
	writeJSON(w, status, map[string]string{
		"error":   string(code),
		"message": message,
	})
}
