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
)

type Server struct {
	registry           modelregistry.Registry
	benchmarkRecorder  benchmarks.Recorder
	layerTransport     layersplit.ActivationTransport
	layerTransportKind layersplit.TransportKind
	now                func() time.Time
	mu                 sync.RWMutex
	nodes              map[string]cluster.NodeRecord
}

type Option func(*Server)

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
	return mux
}
