package coordinator

import (
	"net/http"

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
