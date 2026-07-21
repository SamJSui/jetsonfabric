package facade

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/election"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	PathClusterMembers  = "/v1/cluster/members"
	PathClusterLeader   = "/v1/cluster/leader"
	PathClusterElection = "/v1/cluster/election"
	PathClusterAnnounce = "/v1/cluster/announce"
)

type Config struct {
	SelfID            string
	ClusterToken      string
	Store             *membership.Store
	StaleAfter        time.Duration
	Coordinator       http.Handler
	StageRunner       http.Handler
	RuntimeDeployment http.Handler
}

type Router struct {
	selfID            string
	clusterToken      string
	store             *membership.Store
	staleAfter        time.Duration
	coordinator       http.Handler
	stageRunner       http.Handler
	runtimeDeployment http.Handler
	electionTracker   *election.Tracker
}

type ClusterView struct {
	Leader  *membership.Member  `json:"leader,omitempty"`
	Members []membership.Member `json:"members"`
}

func NewRouter(cfg Config) http.Handler {
	r := newRouter(cfg)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", r.handleHealth)
	mux.HandleFunc("GET "+PathClusterMembers, r.handleMembers)
	mux.HandleFunc("GET "+PathClusterLeader, r.handleLeader)
	mux.HandleFunc("GET "+PathClusterElection, r.handleElection)
	mux.HandleFunc("POST "+PathClusterAnnounce, r.handleAnnounce)
	mux.HandleFunc(api.RouteLayerSplitStage, r.handleStageRun)
	mux.HandleFunc(api.RouteRuntimeDeploymentStatus, r.handleRuntimeDeployment)
	mux.HandleFunc(api.RouteRuntimeDeploymentLoad, r.handleRuntimeDeployment)
	mux.HandleFunc(api.RouteRuntimeDeploymentActivate, r.handleRuntimeDeployment)
	mux.HandleFunc(api.RouteRuntimeDeploymentUnload, r.handleRuntimeDeployment)
	mux.HandleFunc("/", r.handleCoordinator)
	return mux
}

func newRouter(cfg Config) *Router {
	return &Router{
		selfID:            cfg.SelfID,
		clusterToken:      strings.TrimSpace(cfg.ClusterToken),
		store:             cfg.Store,
		staleAfter:        cfg.StaleAfter,
		coordinator:       cfg.Coordinator,
		stageRunner:       cfg.StageRunner,
		runtimeDeployment: cfg.RuntimeDeployment,
		electionTracker:   election.NewTracker(election.DefaultLease(cfg.StaleAfter)),
	}
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

func (r *Router) handleStageRun(w http.ResponseWriter, req *http.Request) {
	if r.stageRunner == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "runtime_stage_gateway_unavailable",
			"message": "this node has no runtime stage gateway configured",
		})
		return
	}
	r.stageRunner.ServeHTTP(w, req)
}

func (r *Router) handleRuntimeDeployment(w http.ResponseWriter, req *http.Request) {
	if r.runtimeDeployment == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "runtime_deployment_gateway_unavailable",
			"message": "this node has no runtime deployment gateway configured",
		})
		return
	}
	if req.Method != http.MethodGet {
		leader, ok := r.leader()
		if !ok {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error":   "coordinator_unavailable",
				"message": "runtime lifecycle writes require an elected coordinator",
			})
			return
		}
		if r.clusterToken == "" {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error":   "coordinator_auth_unconfigured",
				"message": "runtime lifecycle writes require JETSONFABRIC_CLUSTER_TOKEN on every node",
			})
			return
		}
		providedToken := req.Header.Get(api.HeaderClusterToken)
		if strings.TrimSpace(req.Header.Get(api.HeaderCoordinatorNodeID)) != leader.NodeID ||
			subtle.ConstantTimeCompare([]byte(providedToken), []byte(r.clusterToken)) != 1 {
			writeJSON(w, http.StatusForbidden, map[string]string{
				"error":   "coordinator_authentication_required",
				"message": "runtime lifecycle writes require the elected coordinator and cluster token",
			})
			return
		}
	}
	r.runtimeDeployment.ServeHTTP(w, req)
}

func (r *Router) handleCoordinator(w http.ResponseWriter, req *http.Request) {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "leader_unavailable",
			"message": "no healthy leader-eligible node is currently known",
		})
		return
	}
	if leader.NodeID == r.selfID {
		r.serveLocalCoordinator(w, req)
		return
	}
	proxyToLeader(w, req, leader.APIURL)
}

func (r *Router) serveLocalCoordinator(w http.ResponseWriter, req *http.Request) {
	if r.coordinator == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "coordinator_unavailable",
			"message": "this node is leader but has no coordinator router configured",
		})
		return
	}
	r.coordinator.ServeHTTP(w, req)
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
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "leader_proxy_failed", "message": err.Error()})
	}
	proxy.ServeHTTP(w, req)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
