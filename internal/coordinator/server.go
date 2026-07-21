package coordinator

import (
	"net/http"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

type MemberSource interface {
	List() []membership.Member
}

type Server struct {
	nodeID            string
	clusterToken      string
	registry          modelregistry.Registry
	benchmarkRecorder benchmarks.Recorder
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	clusterPlanPolicy clusterplan.Policy
	now               func() time.Time
	deployments       *deploymentState
	deploymentClient  runtimebridge.DeploymentClient
	generationClient  runtimebridge.GenerationClient
	transitionTimeout time.Duration
	cleanupTimeout    time.Duration
	reconcileInterval time.Duration
	isLeader          func(time.Time) bool
	reconcileMu       sync.Mutex
	reconcileCh       chan struct{}
}

type Option func(*Server)

func WithBenchmarkRecorder(recorder benchmarks.Recorder) Option {
	return func(s *Server) {
		s.benchmarkRecorder = recorder
	}
}

func WithMembershipSource(source MemberSource, staleAfter time.Duration) Option {
	return func(s *Server) {
		s.memberSource = source
		s.memberStaleAfter = staleAfter
	}
}

func WithClusterPlanPolicy(policy clusterplan.Policy) Option {
	return func(s *Server) {
		s.clusterPlanPolicy = policy
	}
}

func WithClock(now func() time.Time) Option {
	return func(s *Server) {
		s.now = now
	}
}

func WithDeploymentClient(client runtimebridge.DeploymentClient) Option {
	return func(s *Server) {
		s.deploymentClient = client
	}
}

func WithGenerationClient(client runtimebridge.GenerationClient) Option {
	return func(s *Server) {
		s.generationClient = client
	}
}

func WithNodeID(nodeID string) Option {
	return func(s *Server) {
		s.nodeID = nodeID
	}
}

func WithLeadership(check func(time.Time) bool) Option {
	return func(s *Server) {
		s.isLeader = check
	}
}

func WithDeploymentTimeouts(transition, cleanup time.Duration) Option {
	return func(s *Server) {
		s.transitionTimeout = transition
		s.cleanupTimeout = cleanup
	}
}

func WithReconcileInterval(interval time.Duration) Option {
	return func(s *Server) {
		s.reconcileInterval = interval
	}
}

func WithClusterToken(token string) Option {
	return func(s *Server) {
		s.clusterToken = token
	}
}

func NewServer(registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		registry:          registry,
		benchmarkRecorder: benchmarks.NoopRecorder{},
		now:               func() time.Time { return time.Now().UTC() },
		deployments:       newDeploymentState(),
		transitionTimeout: deploymentSwitchTimeout,
		cleanupTimeout:    deploymentCleanupTimeout,
		reconcileInterval: 5 * time.Second,
		reconcileCh:       make(chan struct{}, 1),
	}
	for _, opt := range opts {
		opt(server)
	}
	server.applyDefaults()
	return server
}

func (s *Server) applyDefaults() {
	if s.benchmarkRecorder == nil {
		s.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if s.now == nil {
		s.now = func() time.Time { return time.Now().UTC() }
	}
	if s.deployments == nil {
		s.deployments = newDeploymentState()
	}
	if s.transitionTimeout <= 0 {
		s.transitionTimeout = deploymentSwitchTimeout
	}
	if s.cleanupTimeout <= 0 {
		s.cleanupTimeout = deploymentCleanupTimeout
	}
	if s.reconcileInterval <= 0 {
		s.reconcileInterval = 5 * time.Second
	}
	if s.reconcileCh == nil {
		s.reconcileCh = make(chan struct{}, 1)
	}
	if s.deploymentClient == nil {
		s.deploymentClient = runtimebridge.NewHTTPDeploymentClient(runtimebridge.HTTPDeploymentClientConfig{
			Timeout:           10 * time.Minute,
			CoordinatorNodeID: s.nodeID,
			ClusterToken:      s.clusterToken,
		})
	}
	if s.generationClient == nil {
		s.generationClient = runtimebridge.NewHTTPGenerationClient(runtimebridge.HTTPGenerationClientConfig{
			CoordinatorNodeID: s.nodeID,
			ClusterToken:      s.clusterToken,
		})
	}
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteModels, s.handleModels)
	mux.HandleFunc(api.RoutePreview, s.handleRoutePreview)
	mux.HandleFunc(api.RouteLayerSplitRun, s.handleLayerSplitRun)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	mux.HandleFunc(api.RouteDeploymentStatus, s.handleDeploymentStatus)
	mux.HandleFunc(api.RouteDeploymentSwitch, s.handleDeploymentSwitch)
	return mux
}
