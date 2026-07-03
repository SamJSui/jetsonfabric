package control

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func (s *Server) handleNodes(w http.ResponseWriter, _ *http.Request) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)
	writeJSON(w, http.StatusOK, map[string]any{"nodes": nodes})
}

func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, errorUnauthorized, "agent join token is missing or invalid")
		return
	}
	defer r.Body.Close()
	var heartbeat cluster.HeartbeatRequest
	if err := json.NewDecoder(r.Body).Decode(&heartbeat); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	if heartbeat.NodeName == "" {
		writeError(w, http.StatusBadRequest, errorMissingNodeName, "node_name is required")
		return
	}
	record := cluster.NodeRecord{
		NodeName:     heartbeat.NodeName,
		Hostname:     defaultString(heartbeat.Hostname, heartbeat.NodeName),
		Arch:         defaultString(heartbeat.Arch, "unknown"),
		OS:           heartbeat.OS,
		Capabilities: defaultMap(heartbeat.Capabilities),
		Metrics:      defaultMap(heartbeat.Metrics),
		Engines:      heartbeat.Engines,
		LastSeen:     s.now(),
	}
	s.mu.Lock()
	s.nodes[record.NodeName] = record
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered", "node": record})
}

func (s *Server) authorized(r *http.Request) bool {
	if s.joinToken == "" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.joinToken
}

func defaultString(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func defaultMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}
