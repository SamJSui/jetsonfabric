package control

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
	joinToken          string
	registry           modelregistry.Registry
	backendFactory     BackendFactory
	benchmarkRecorder  benchmarks.Recorder
	layerTransport     layersplit.ActivationTransport
	layerTransportKind layersplit.TransportKind
	now                func() time.Time
	mu                 sync.RWMutex
	nodes              map[string]cluster.NodeRecord
}

type BackendFactory func(cluster.RuntimeBackend) (runtimeclient.ChatBackend, error)

type Option func(*Server)

func WithBackendFactory(factory BackendFactory) Option {
	return func(s *Server) {
		s.backendFactory = factory
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

func NewServer(joinToken string, registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		joinToken:          joinToken,
		registry:           registry,
		backendFactory:     defaultBackendFactory,
		benchmarkRecorder:  benchmarks.NoopRecorder{},
		layerTransportKind: layersplit.TransportHTTP,
		now:                func() time.Time { return time.Now().UTC() },
		nodes:              make(map[string]cluster.NodeRecord),
	}
	for _, opt := range opts {
		opt(server)
	}
	if server.backendFactory == nil {
		server.backendFactory = defaultBackendFactory
	}
	if server.benchmarkRecorder == nil {
		server.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if server.layerTransport == nil {
		transport, err := layersplit.NewTransport(server.layerTransportKind)
		if err != nil {
			panic(err)
		}
		server.layerTransport = transport
	}
	if server.now == nil {
		server.now = func() time.Time { return time.Now().UTC() }
	}
	return server
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteNodes, s.handleNodes)
	mux.HandleFunc(api.RouteModels, s.handleModels)
	mux.HandleFunc(api.RoutePreview, s.handleRoutePreview)
	mux.HandleFunc(api.RouteLayerSplitPlan, s.handleLayerSplitPlan)
	mux.HandleFunc(api.RouteLayerSplitChat, s.handleLayerSplitCompletions)
	mux.HandleFunc(api.RouteAgentHeartbeat, s.handleHeartbeat)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	return mux
}
