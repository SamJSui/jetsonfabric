package facade

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterauth"
	"github.com/SamJSui/jetsonfabric/internal/election"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const maxAnnouncementBytes = 1 << 20

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
	payload, ok := readAnnouncementPayload(w, req)
	if !ok {
		return
	}
	if r.clusterToken != "" && !r.authorizeAnnouncement(w, req, payload) {
		return
	}

	member, ok := r.decodeAnnouncedMember(w, payload)
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
	r.writeAnnouncementResponse(w, req, r.clusterView())
}

func readAnnouncementPayload(w http.ResponseWriter, req *http.Request) ([]byte, bool) {
	payload, err := io.ReadAll(io.LimitReader(req.Body, maxAnnouncementBytes+1))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_member", "message": "could not read member record"})
		return nil, false
	}
	if len(payload) > maxAnnouncementBytes {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "invalid_member", "message": "member record exceeds 1 MiB"})
		return nil, false
	}
	return payload, true
}

func (r *Router) authorizeAnnouncement(w http.ResponseWriter, req *http.Request, payload []byte) bool {
	if clusterauth.VerifyAnnouncementRequest(
		r.clusterToken,
		req.Header.Get(clusterauth.HeaderAnnouncementTimestamp),
		req.Header.Get(clusterauth.HeaderAnnouncementNonce),
		req.Header.Get(clusterauth.HeaderAnnouncementSignature),
		payload,
		time.Now().UTC(),
	) {
		return true
	}
	writeJSON(w, http.StatusForbidden, map[string]string{
		"error":   "cluster_authentication_required",
		"message": "cluster announcements require a valid cluster signature",
	})
	return false
}

func (r *Router) writeAnnouncementResponse(w http.ResponseWriter, req *http.Request, view ClusterView) {
	payload, err := json.Marshal(view)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error":   "encode_cluster_view_failed",
			"message": "could not encode cluster view",
		})
		return
	}
	if r.clusterToken != "" {
		timestamp, signature := clusterauth.SignAnnouncementResponse(
			r.clusterToken,
			time.Now().UTC(),
			req.Header.Get(clusterauth.HeaderAnnouncementNonce),
			payload,
		)
		w.Header().Set(clusterauth.HeaderAnnouncementResponseTimestamp, timestamp)
		w.Header().Set(clusterauth.HeaderAnnouncementResponseSignature, signature)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(payload)
}

func (r *Router) decodeAnnouncedMember(w http.ResponseWriter, payload []byte) (membership.Member, bool) {
	var member membership.Member
	if err := json.NewDecoder(bytes.NewReader(payload)).Decode(&member); err != nil {
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
