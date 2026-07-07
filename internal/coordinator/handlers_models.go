package coordinator

import (
	"net/http"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/routing"
)

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
