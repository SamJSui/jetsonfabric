package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/benchmarks"
	"github.com/SamJSui/JetsonMesh/internal/chat"
	"github.com/SamJSui/JetsonMesh/internal/cluster"
	"github.com/SamJSui/JetsonMesh/internal/modelregistry"
	"github.com/SamJSui/JetsonMesh/internal/routing"
	"github.com/SamJSui/JetsonMesh/internal/runtimeclient"
)

type Server struct {
	joinToken         string
	registry          modelregistry.Registry
	backendFactory    BackendFactory
	benchmarkRecorder benchmarks.Recorder
	now               func() time.Time
	mu                sync.RWMutex
	nodes             map[string]cluster.NodeRecord
}

type BackendFactory func(cluster.RuntimeBackend) runtimeclient.ChatBackend

type Option func(*Server)

func WithBackendFactory(factory BackendFactory) Option {
	return func(s *Server) {
		s.backendFactory = factory
	}
}

func WithBenchmarkRecorder(recorder benchmarks.Recorder) Option {
	return func(s *Server) {
		s.benchmarkRecorder = recorder
	}
}

func WithClock(now func() time.Time) Option {
	return func(s *Server) {
		s.now = now
	}
}

func NewServer(joinToken string, registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		joinToken:         joinToken,
		registry:          registry,
		backendFactory:    defaultBackendFactory,
		benchmarkRecorder: benchmarks.NoopRecorder{},
		now:               func() time.Time { return time.Now().UTC() },
		nodes:             make(map[string]cluster.NodeRecord),
	}
	for _, opt := range opts {
		opt(server)
	}
	if server.backendFactory == nil {
		server.backendFactory = defaultBackendFactory
	}
	if server.benchmarkRecorder == nil {
		server.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if server.now == nil {
		server.now = func() time.Time { return time.Now().UTC() }
	}
	return server
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /v1/nodes", s.handleNodes)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /v1/routes/preview", s.handleRoutePreview)
	mux.HandleFunc("POST /v1/agents/heartbeat", s.handleHeartbeat)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "jetsonmesh-control"})
}

func (s *Server) handleNodes(w http.ResponseWriter, _ *http.Request) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })
	writeJSON(w, http.StatusOK, map[string]any{"nodes": nodes})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.registry)
}

func (s *Server) handleRoutePreview(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("model")
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeJSON(w, http.StatusOK, routing.UnknownModel(modelID))
		return
	}
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	writeJSON(w, http.StatusOK, routing.Preview(model, nodes))
}

func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
		return
	}
	defer r.Body.Close()
	var heartbeat cluster.HeartbeatRequest
	if err := json.NewDecoder(r.Body).Decode(&heartbeat); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_json"})
		return
	}
	if heartbeat.NodeID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing_node_id"})
		return
	}
	record := cluster.NodeRecord{
		NodeID:       heartbeat.NodeID,
		Hostname:     fallback(heartbeat.Hostname, heartbeat.NodeID),
		Arch:         fallback(heartbeat.Arch, "unknown"),
		OS:           fallback(heartbeat.OS, "unknown"),
		Capabilities: fallbackMap(heartbeat.Capabilities),
		Metrics:      fallbackMap(heartbeat.Metrics),
		Backends:     heartbeat.Backends,
		LastSeen:     s.now(),
	}
	s.mu.Lock()
	s.nodes[record.NodeID] = record
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered", "node": record})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req chat.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", "request body must be valid JSON")
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "missing_model", "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "missing_messages", "messages must contain at least one message")
		return
	}
	model, ok := s.registry.Find(req.Model)
	if !ok {
		writeError(w, http.StatusBadRequest, "unknown_model", fmt.Sprintf("model %q is not in the registry", req.Model))
		return
	}
	node, backend, err := s.selectSingleNodeBackend(model)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "no_single_node_route", err.Error())
		return
	}

	resp, stats, err := s.backendFactory(backend).Complete(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "backend_request_failed", err.Error())
		return
	}
	resp.Route = &chat.RouteMetadata{
		Mode:        "single_node",
		NodeID:      node.NodeID,
		BackendID:   backend.ID,
		BackendKind: backend.Kind,
		LatencyMS:   stats.Latency.Milliseconds(),
	}

	record := benchmarks.Record{
		Timestamp:    s.now(),
		ModelID:      model.ID,
		NodeID:       node.NodeID,
		RouteMode:    "single_node",
		BackendID:    backend.ID,
		BackendKind:  backend.Kind,
		LatencyMS:    stats.Latency.Milliseconds(),
		OutputTokens: stats.OutputTokens,
		TokensPerSec: stats.TokensPerSec,
		MemoryGB:     optionalFloat(node.Capabilities, "memory_gb"),
		TemperatureC: optionalFloat(node.Metrics, "temperature_c"),
	}
	if err := s.benchmarkRecorder.Record(r.Context(), record); err != nil {
		writeError(w, http.StatusInternalServerError, "benchmark_record_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) authorized(r *http.Request) bool {
	if s.joinToken == "" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.joinToken
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, code string, message string) {
	writeJSON(w, status, map[string]string{
		"error":   code,
		"message": message,
	})
}

func fallback(value string, defaultValue string) string {
	if strings.TrimSpace(value) == "" {
		return defaultValue
	}
	return value
}

func fallbackMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

func defaultBackendFactory(backend cluster.RuntimeBackend) runtimeclient.ChatBackend {
	return runtimeclient.NewOpenAIClient(backend.BaseURL, 60*time.Second)
}

func (s *Server) selectSingleNodeBackend(model cluster.ModelProfile) (cluster.NodeRecord, cluster.RuntimeBackend, error) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })

	preview := routing.Preview(model, nodes)
	placementByNode := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placementByNode[placement.NodeID] = placement
	}
	for _, node := range nodes {
		placement := placementByNode[node.NodeID]
		if !placement.Valid {
			continue
		}
		for _, backend := range node.Backends {
			if backendCompatible(model, backend) {
				return node, backend, nil
			}
		}
	}
	if len(nodes) == 0 {
		return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no online nodes")
	}
	return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no compatible backend for model %q", model.ID)
}

func backendCompatible(model cluster.ModelProfile, backend cluster.RuntimeBackend) bool {
	if strings.TrimSpace(backend.BaseURL) == "" {
		return false
	}
	if !backend.OpenAICompatible {
		return false
	}
	if len(backend.Models) > 0 {
		for _, modelID := range backend.Models {
			if modelID == model.ID {
				return true
			}
		}
		return false
	}
	return backend.Kind == model.Runtime
}

func optionalFloat(values map[string]any, key string) *float64 {
	value, ok := values[key]
	if !ok {
		return nil
	}
	var output float64
	switch typed := value.(type) {
	case float64:
		output = typed
	case float32:
		output = float64(typed)
	case int:
		output = float64(typed)
	case int64:
		output = float64(typed)
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return nil
		}
		output = parsed
	default:
		return nil
	}
	return &output
}
