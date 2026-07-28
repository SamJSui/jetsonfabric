package facade

import "net/http"

func (r *Router) handleHealth(w http.ResponseWriter, _ *http.Request) {
	leader, _ := r.leader()
	self, _ := r.store.Get(r.selfID)
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"service": "jetsonfabric-node",
		"node_id": self.NodeID,
		"leader":  leader,
	})
}
