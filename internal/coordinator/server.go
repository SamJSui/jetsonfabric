package coordinator

import (
	"net/http"
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
	clusterToken      string
	registry          modelregistry.Registry
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	clusterPlanPolicy clusterplan.Policy
	now               func() time.Time
	deployments       *DeploymentController
	generations       *GenerationController
}

type serverConfig struct {
	nodeID            string
	clusterToken      string
	benchmarkRecorder benchmarks.Recorder
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	clusterPlanPolicy clusterplan.Policy
	now               func() time.Time
	deploymentClient  runtimebridge.DeploymentClient
	generationClient  runtimebridge.GenerationClient
	transitionTimeout time.Duration
	cleanupTimeout    time.Duration
	reconcileInterval time.Duration
	isLeader          func(time.Time) bool
}

type Option func(*serverConfig)

func WithBenchmarkRecorder(recorder benchmarks.Recorder) Option {
	return func(cfg *serverConfig) {
		cfg.benchmarkRecorder = recorder
	}
}

func WithMembershipSource(source MemberSource, staleAfter time.Duration) Option {
	return func(cfg *serverConfig) {
		cfg.memberSource = source
		cfg.memberStaleAfter = staleAfter
	}
}

func WithClusterPlanPolicy(policy clusterplan.Policy) Option {
	return func(cfg *serverConfig) {
		cfg.clusterPlanPolicy = policy
	}
}

func WithClock(now func() time.Time) Option {
	return func(cfg *serverConfig) {
		cfg.now = now
	}
}

func WithDeploymentClient(client runtimebridge.DeploymentClient) Option {
	return func(cfg *serverConfig) {
		cfg.deploymentClient = client
	}
}

func WithGenerationClient(client runtimebridge.GenerationClient) Option {
	return func(cfg *serverConfig) {
		cfg.generationClient = client
	}
}

func WithNodeID(nodeID string) Option {
	return func(cfg *serverConfig) {
		cfg.nodeID = nodeID
	}
}

func WithLeadership(check func(time.Time) bool) Option {
	return func(cfg *serverConfig) {
		cfg.isLeader = check
	}
}

func WithDeploymentTimeouts(transition, cleanup time.Duration) Option {
	return func(cfg *serverConfig) {
		cfg.transitionTimeout = transition
		cfg.cleanupTimeout = cleanup
	}
}

func WithReconcileInterval(interval time.Duration) Option {
	return func(cfg *serverConfig) {
		cfg.reconcileInterval = interval
	}
}

func WithClusterToken(token string) Option {
	return func(cfg *serverConfig) {
		cfg.clusterToken = token
	}
}

func NewServer(registry modelregistry.Registry, opts ...Option) *Server {
	cfg := serverConfig{
		benchmarkRecorder: benchmarks.NoopRecorder{},
		now:               func() time.Time { return time.Now().UTC() },
		transitionTimeout: deploymentSwitchTimeout,
		cleanupTimeout:    deploymentCleanupTimeout,
		reconcileInterval: 5 * time.Second,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	cfg.applyDefaults()
	server := &Server{
		clusterToken:      cfg.clusterToken,
		registry:          registry,
		memberSource:      cfg.memberSource,
		memberStaleAfter:  cfg.memberStaleAfter,
		clusterPlanPolicy: cfg.clusterPlanPolicy,
		now:               cfg.now,
	}
	server.deployments = newDeploymentController(deploymentControllerConfig{
		registry:          server.registry,
		memberSource:      server.memberSource,
		memberStaleAfter:  server.memberStaleAfter,
		planPolicy:        server.clusterPlanPolicy,
		now:               server.now,
		runtimeClient:     cfg.deploymentClient,
		transitionTimeout: cfg.transitionTimeout,
		cleanupTimeout:    cfg.cleanupTimeout,
		reconcileInterval: cfg.reconcileInterval,
		isLeader:          cfg.isLeader,
	})
	server.generations = newGenerationController(
		server.registry,
		server.memberSource,
		server.memberStaleAfter,
		server.now,
		server.deployments,
		cfg.generationClient,
	)
	return server
}

func (cfg *serverConfig) applyDefaults() {
	if cfg.benchmarkRecorder == nil {
		cfg.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if cfg.now == nil {
		cfg.now = func() time.Time { return time.Now().UTC() }
	}
	if cfg.transitionTimeout <= 0 {
		cfg.transitionTimeout = deploymentSwitchTimeout
	}
	if cfg.cleanupTimeout <= 0 {
		cfg.cleanupTimeout = deploymentCleanupTimeout
	}
	if cfg.reconcileInterval <= 0 {
		cfg.reconcileInterval = 5 * time.Second
	}
	if cfg.deploymentClient == nil {
		cfg.deploymentClient = runtimebridge.NewHTTPDeploymentClient(runtimebridge.HTTPDeploymentClientConfig{
			Timeout:           10 * time.Minute,
			CoordinatorNodeID: cfg.nodeID,
			ClusterToken:      cfg.clusterToken,
		})
	}
	if cfg.generationClient == nil {
		cfg.generationClient = runtimebridge.NewHTTPGenerationClient(runtimebridge.HTTPGenerationClientConfig{
			CoordinatorNodeID: cfg.nodeID,
			ClusterToken:      cfg.clusterToken,
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
