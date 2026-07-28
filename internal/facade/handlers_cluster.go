package facade

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/election"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type ClusterView struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

func (r *Router) handleMembers(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, r.clusterView())
}

func (r *Router) handleElection(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, r.electionResult(time.Now().UTC()))
}

func (r *Router) handleLeader(w http.ResponseWriter, _ *http.Request) {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "leader_unavailable",
			"message": "no healthy leader-eligible node is currently known",
		})
		return
	}
	writeJSON(w, http.StatusOK, leader)
}

func (r *Router) handleAnnounce(w http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	member, ok := r.decodeAnnouncedMember(w, req)
	if !ok {
		return
	}
	if !r.sameCluster(member) {
		writeJSON(w, http.StatusForbidden, map[string]string{
			"error":   "cluster_mismatch",
			"message": "announced member belongs to a different cluster",
		})
		return
	}

	member.LastSeen = time.Now().UTC()
	r.store.Upsert(member)
	writeJSON(w, http.StatusOK, r.clusterView())
}

func (r *Router) decodeAnnouncedMember(w http.ResponseWriter, req *http.Request) (membership.Member, bool) {
	var member membership.Member
	if err := json.NewDecoder(req.Body).Decode(&member); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_member", "message": "request body must be a valid member record"})
		return membership.Member{}, false
	}
	member = membership.Normalize(member)
	if !member.Valid() {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_member", "message": "cluster_id, node_id, and api_url are required"})
		return membership.Member{}, false
	}
	return member, true
}

func (r *Router) sameCluster(member membership.Member) bool {
	self, ok := r.store.Get(r.selfID)
	return !ok || self.ClusterID == "" || member.ClusterID == self.ClusterID
}

func (r *Router) clusterView() ClusterView {
	leader, ok := r.leader()
	view := ClusterView{Members: r.visibleMembers(time.Now().UTC())}
	if ok {
		view.Leader = &leader
	}
	return view
}

func (r *Router) leader() (membership.Member, bool) {
	result := r.electionResult(time.Now().UTC())
	if result.Leader == nil {
		return membership.Member{}, false
	}
	return *result.Leader, true
}

func (r *Router) electionResult(now time.Time) election.Result {
	return r.electionTracker.Explain(now, r.store.List(), r.staleAfter)
}

func (r *Router) visibleMembers(now time.Time) []membership.Member {
	members := r.store.List()
	visible := make([]membership.Member, 0, len(members))
	for _, member := range members {
		if !member.IsStale(now, r.staleAfter) {
			visible = append(visible, member)
		}
	}
	return visible
}
