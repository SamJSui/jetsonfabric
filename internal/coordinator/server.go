package coordinator

import (
	"net/http"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
)

type MemberSource interface {
	List() []membership.Member
}

type Server struct {
	registry          modelregistry.Registry
	benchmarkRecorder benchmarks.Recorder
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	clusterPlanPolicy clusterplan.Policy
	now               func() time.Time
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

func NewServer(registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		registry:          registry,
		benchmarkRecorder: benchmarks.NoopRecorder{},
		now:               func() time.Time { return time.Now().UTC() },
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
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteModels, s.handleModels)
	mux.HandleFunc(api.RoutePreview, s.handleRoutePreview)
	mux.HandleFunc(api.RouteLayerSplitRun, s.handleLayerSplitRun)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	return mux
}
