package facade

import (
	"net/http"
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
	RuntimeGeneration http.Handler
}

type Router struct {
	selfID            string
	clusterToken      string
	store             *membership.Store
	staleAfter        time.Duration
	coordinator       http.Handler
	stageRunner       http.Handler
	runtimeDeployment http.Handler
	runtimeGeneration http.Handler
	electionTracker   *election.Tracker
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
	mux.HandleFunc(api.RouteRuntimeDeploymentDrain, r.handleRuntimeDeployment)
	mux.HandleFunc(api.RouteRuntimeDeploymentUnload, r.handleRuntimeDeployment)
	mux.HandleFunc(api.RouteRuntimeGeneration, r.handleRuntimeGeneration)
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
		runtimeGeneration: cfg.RuntimeGeneration,
		electionTracker:   election.NewTracker(election.DefaultLease(cfg.StaleAfter)),
	}
}
