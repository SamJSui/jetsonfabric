package control

import (
	"encoding/json"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
	"github.com/SamJSui/JetsonMesh/internal/modelregistry"
	"github.com/SamJSui/JetsonMesh/internal/routing"
)

type Server struct {
	joinToken string
	registry  modelregistry.Registry
	mu        sync.RWMutex
	nodes     map[string]cluster.NodeRecord
}

func NewServer(joinToken string, registry modelregistry.Registry) *Server {
	return &Server{
		joinToken: joinToken,
		registry:  registry,
		nodes:     make(map[string]cluster.NodeRecord),
	}
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
		LastSeen:     time.Now().UTC(),
	}
	s.mu.Lock()
	s.nodes[record.NodeID] = record
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered", "node": record})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{
		"error":   "not_implemented",
		"message": "OpenAI-compatible model routing is scaffolded but no runtime backend is wired yet.",
		"planned_router_inputs": []string{
			"model",
			"latency_budget_ms",
			"quality_floor",
			"node_queue_depth",
			"node_temperature",
			"model_fit",
		},
	})
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
