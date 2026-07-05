package node

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/coordinator"
	"github.com/SamJSui/jetsonfabric/internal/discovery"
	"github.com/SamJSui/jetsonfabric/internal/facade"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimegateway"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

type App struct {
	cfg            Config
	nodeID         string
	startedAt      time.Time
	store          *membership.Store
	discovery      discovery.Source
	mdnsAdvertiser *discovery.MDNSAdvertiser
	server         *http.Server
}

func New(cfg Config) (*App, error) {
	cfg = NormalizeConfig(cfg)
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}

	nodeID, err := LoadOrCreateNodeID(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	app := newApp(cfg, nodeID)

	coordinatorRouter, err := app.coordinatorRouter()
	if err != nil {
		return nil, err
	}
	stageRunner, err := runtimegateway.NewStageProxy(cfg.RuntimeURL)
	if err != nil {
		return nil, fmt.Errorf("create runtime stage gateway: %w", err)
	}

	self := func() membership.Member { return app.selfMember(time.Now().UTC()) }
	app.discovery = app.discoverySource(self)
	app.mdnsAdvertiser = app.newMDNSAdvertiser(self)
	app.server = app.httpServer(coordinatorRouter, stageRunner)
	return app, nil
}

func newApp(cfg Config, nodeID string) *App {
	app := &App{
		cfg:       cfg,
		nodeID:    nodeID,
		startedAt: time.Now().UTC(),
		store:     membership.NewStore(),
	}
	app.store.Upsert(app.selfMember(time.Now().UTC()))
	return app
}

func (a *App) coordinatorRouter() (http.Handler, error) {
	registry, err := modelregistry.Load(a.cfg.ModelsPath)
	if err != nil {
		return nil, fmt.Errorf("load model registry: %w", err)
	}
	server := coordinator.NewServer(
		a.cfg.JoinToken,
		registry,
		coordinator.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(a.cfg.BenchmarksPath)),
	)
	return server.Router(), nil
}

func (a *App) newMDNSAdvertiser(self discovery.SelfFunc) *discovery.MDNSAdvertiser {
	if !a.cfg.DiscoveryEnabled(discovery.ModeMDNS) {
		return nil
	}
	return discovery.NewMDNSAdvertiser(discovery.MDNSConfig{
		ClusterID: a.cfg.ClusterID,
		Service:   a.cfg.MDNSService,
		Domain:    a.cfg.MDNSDomain,
		Port:      a.cfg.AdvertisePort(),
		Self:      self,
	})
}

func (a *App) httpServer(coordinatorRouter http.Handler, stageRunner http.Handler) *http.Server {
	return &http.Server{
		Addr: a.cfg.Listen,
		Handler: facade.NewRouter(facade.Config{
			SelfID:      a.nodeID,
			Store:       a.store,
			StaleAfter:  a.cfg.StaleAfter,
			Coordinator: coordinatorRouter,
			StageRunner: stageRunner,
		}),
		ReadHeaderTimeout: 5 * time.Second,
	}
}

func (a *App) Run(ctx context.Context) error {
	a.startMDNS(ctx)
	go a.discoveryLoop(ctx)

	errCh := make(chan error, 1)
	go a.listen(errCh)

	select {
	case <-ctx.Done():
		return a.shutdown(errCh)
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
}

func (a *App) startMDNS(ctx context.Context) {
	if a.mdnsAdvertiser == nil {
		return
	}
	if err := a.mdnsAdvertiser.Start(ctx); err != nil {
		log.Printf("mDNS advertising disabled: %v", err)
		return
	}
	log.Printf("JetsonFabric node advertising with mDNS service=%s domain=%s", a.cfg.MDNSService, a.cfg.MDNSDomain)
}

func (a *App) listen(errCh chan<- error) {
	log.Printf(
		"JetsonFabric node listening on http://%s advertised=%s cluster=%s node_id=%s discovery=%v",
		a.cfg.Listen,
		a.cfg.APIURL,
		a.cfg.ClusterID,
		a.nodeID,
		a.cfg.DiscoveryModes,
	)
	errCh <- a.server.ListenAndServe()
}

func (a *App) shutdown(errCh <-chan error) error {
	if a.mdnsAdvertiser != nil {
		a.mdnsAdvertiser.Shutdown()
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := a.server.Shutdown(shutdownCtx); err != nil {
		return err
	}
	if err := <-errCh; err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

func (a *App) discoverySource(self discovery.SelfFunc) discovery.Source {
	announcer := discovery.NewAnnounceClient(self)
	sources := make([]discovery.Source, 0, 2)
	if a.cfg.DiscoveryEnabled(discovery.ModeStatic) {
		sources = append(sources, discovery.NewStaticSource(a.cfg.Seeds, self))
	}
	if a.cfg.DiscoveryEnabled(discovery.ModeMDNS) {
		sources = append(sources, a.hydratingMDNSSource(announcer))
	}
	return discovery.NewMultiSource(sources...)
}

func (a *App) hydratingMDNSSource(announcer *discovery.AnnounceClient) discovery.Source {
	mdnsSource := discovery.NewMDNSSource(discovery.MDNSConfig{
		ClusterID:     a.cfg.ClusterID,
		Service:       a.cfg.MDNSService,
		Domain:        a.cfg.MDNSDomain,
		BrowseTimeout: a.cfg.MDNSBrowseTimeout,
	})
	return discovery.NewHydratingSource(mdnsSource, announcer)
}

func (a *App) discoveryLoop(ctx context.Context) {
	ticker := time.NewTicker(a.cfg.DiscoveryInterval)
	defer ticker.Stop()

	a.refreshMembership(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			a.refreshMembership(ctx)
		}
	}
}

func (a *App) refreshMembership(ctx context.Context) {
	now := time.Now().UTC()
	a.store.Upsert(a.selfMember(now))
	a.store.Prune(now, a.cfg.StaleAfter, a.nodeID)
	if a.discovery == nil {
		return
	}

	members, err := a.discovery.Discover(ctx)
	if err != nil {
		log.Printf("discovery failed: %v", err)
		return
	}
	a.mergeDiscoveredMembers(members, now)
}

func (a *App) mergeDiscoveredMembers(members []membership.Member, now time.Time) {
	for _, member := range members {
		member = membership.Normalize(member)
		if member.NodeID == a.nodeID || member.ClusterID != a.cfg.ClusterID {
			continue
		}
		if member.LastSeen.IsZero() {
			member.LastSeen = now
		}
		a.store.Upsert(member)
	}
}

func (a *App) selfMember(now time.Time) membership.Member {
	snapshot := system.Detect()
	nodeName := a.cfg.NodeName
	if nodeName == "" {
		nodeName = snapshot.Hostname
	}
	if nodeName == "" {
		nodeName, _ = os.Hostname()
	}

	return membership.Member{
		ClusterID:       a.cfg.ClusterID,
		NodeID:          a.nodeID,
		NodeName:        nodeName,
		Hostname:        snapshot.Hostname,
		APIURL:          a.cfg.APIURL,
		RuntimeURL:      a.cfg.RuntimeURL,
		ControlEligible: a.cfg.ControlEligible,
		ControlPriority: a.cfg.ControlPriority,
		Arch:            snapshot.Arch,
		OS:              snapshot.OS,
		Capabilities:    snapshot.Capabilities,
		Metrics:         snapshot.Metrics,
		StartedAt:       a.startedAt,
		LastSeen:        now,
	}
}
