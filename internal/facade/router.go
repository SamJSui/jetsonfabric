package facade

import (
	"encoding/json"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/election"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	PathClusterMembers  = "/v1/cluster/members"
	PathClusterLeader   = "/v1/cluster/leader"
	PathClusterAnnounce = "/v1/cluster/announce"
)

type Config struct {
	SelfID      string
	Store       *membership.Store
	StaleAfter  time.Duration
	Coordinator http.Handler
}

type Router struct {
	selfID      string
	store       *membership.Store
	staleAfter  time.Duration
	coordinator http.Handler
}

type ClusterView struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

func NewRouter(cfg Config) http.Handler {
	r := &Router{
		selfID:      cfg.SelfID,
		store:       cfg.Store,
		staleAfter:  cfg.StaleAfter,
		coordinator: cfg.Coordinator,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", r.handleHealth)
	mux.HandleFunc("GET "+PathClusterMembers, r.handleMembers)
	mux.HandleFunc("GET "+PathClusterLeader, r.handleLeader)
	mux.HandleFunc("POST "+PathClusterAnnounce, r.handleAnnounce)
	mux.HandleFunc("/", r.handleCoordinator)
	return mux
}

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

func (r *Router) handleMembers(w http.ResponseWriter, _ *http.Request) {
	view := r.clusterView()
	writeJSON(w, http.StatusOK, view)
}

func (r *Router) handleLeader(w http.ResponseWriter, _ *http.Request) {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "leader_unavailable",
			"message": "no healthy control-eligible node is currently known",
		})
		return
	}
	writeJSON(w, http.StatusOK, leader)
}

func (r *Router) handleAnnounce(w http.ResponseWriter, req *http.Request) {
	defer req.Body.Close()

	var member membership.Member
	if err := json.NewDecoder(req.Body).Decode(&member); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error":   "invalid_member",
			"message": "request body must be a valid member record",
		})
		return
	}
	member = membership.Normalize(member)
	if !member.Valid() {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error":   "invalid_member",
			"message": "cluster_id, node_id, and api_url are required",
		})
		return
	}

	self, ok := r.store.Get(r.selfID)
	if ok && self.ClusterID != "" && member.ClusterID != self.ClusterID {
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

func (r *Router) handleCoordinator(w http.ResponseWriter, req *http.Request) {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "leader_unavailable",
			"message": "no healthy control-eligible node is currently known",
		})
		return
	}

	if leader.NodeID == r.selfID {
		if r.coordinator == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error":   "coordinator_unavailable",
				"message": "this node is leader but has no coordinator router configured",
			})
			return
		}
		r.coordinator.ServeHTTP(w, req)
		return
	}

	proxyToLeader(w, req, leader.APIURL)
}

func (r *Router) clusterView() ClusterView {
	members := r.activeMembers(time.Now().UTC())
	leader, ok := election.ElectLeader(time.Now().UTC(), members, r.staleAfter)
	view := ClusterView{Members: members}
	if ok {
		view.Leader = &leader
	}
	return view
}

func (r *Router) leader() (membership.Member, bool) {
	now := time.Now().UTC()
	return election.ElectLeader(now, r.activeMembers(now), r.staleAfter)
}

func (r *Router) activeMembers(now time.Time) []membership.Member {
	members := r.store.List()
	active := make([]membership.Member, 0, len(members))
	for _, member := range members {
		if member.IsStale(now, r.staleAfter) {
			continue
		}
		active = append(active, member)
	}
	return active
}

func proxyToLeader(w http.ResponseWriter, req *http.Request, leaderURL string) {
	target, err := url.Parse(leaderURL)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"error":   "invalid_leader_url",
			"message": err.Error(),
		})
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"error":   "leader_proxy_failed",
			"message": err.Error(),
		})
	}
	proxy.ServeHTTP(w, req)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
