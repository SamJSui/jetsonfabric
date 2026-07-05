package coordinator

import (
	"net/http"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

type Server struct {
	registry           modelregistry.Registry
	engineFactory      EngineFactory
	benchmarkRecorder  benchmarks.Recorder
	layerTransport     layersplit.ActivationTransport
	layerTransportKind layersplit.TransportKind
	now                func() time.Time
	mu                 sync.RWMutex
	nodes              map[string]cluster.NodeRecord
}

type EngineFactory func(cluster.EngineEndpoint) (runtimeclient.ChatBackend, error)

type Option func(*Server)

func WithEngineFactory(factory EngineFactory) Option {
	return func(s *Server) {
		s.engineFactory = factory
	}
}

func WithBenchmarkRecorder(recorder benchmarks.Recorder) Option {
	return func(s *Server) {
		s.benchmarkRecorder = recorder
	}
}

func WithLayerTransport(kind layersplit.TransportKind, transport layersplit.ActivationTransport) Option {
	return func(s *Server) {
		s.layerTransportKind = kind
		s.layerTransport = transport
	}
}

func WithClock(now func() time.Time) Option {
	return func(s *Server) {
		s.now = now
	}
}

func NewServer(registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		registry:           registry,
		engineFactory:      defaultEngineFactory,
		benchmarkRecorder:  benchmarks.NoopRecorder{},
		layerTransportKind: layersplit.TransportHTTP,
		now:                func() time.Time { return time.Now().UTC() },
		nodes:              make(map[string]cluster.NodeRecord),
	}
	for _, opt := range opts {
		opt(server)
	}
	server.applyDefaults()
	return server
}

func (s *Server) applyDefaults() {
	if s.engineFactory == nil {
		s.engineFactory = defaultEngineFactory
	}
	if s.benchmarkRecorder == nil {
		s.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if s.now == nil {
		s.now = func() time.Time { return time.Now().UTC() }
	}
	s.ensureLayerTransport()
}

func (s *Server) ensureLayerTransport() {
	if s.layerTransport != nil {
		return
	}
	transport, err := layersplit.NewTransport(s.layerTransportKind)
	if err != nil {
		panic(err)
	}
	s.layerTransport = transport
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteModels, s.handleModels)
	mux.HandleFunc(api.RoutePreview, s.handleRoutePreview)
	mux.HandleFunc(api.RouteLayerSplitPlan, s.handleLayerSplitPlan)
	mux.HandleFunc(api.RouteLayerSplitChat, s.handleLayerSplitCompletions)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	return mux
}
