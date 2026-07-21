package node

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/coordinator"
	"github.com/SamJSui/jetsonfabric/internal/discovery"
	"github.com/SamJSui/jetsonfabric/internal/facade"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

type App struct {
	cfg                 Config
	nodeID              string
	startedAt           time.Time
	modelArtifactSHA256 string
	store               *membership.Store
	discovery           discovery.Source
	mdnsAdvertiser      *discovery.MDNSAdvertiser
	server              *http.Server
	runtimeSupervisor   *RuntimeSupervisor
}

func New(cfg Config) (*App, error) {
	return buildApp(cfg)
}

func buildApp(cfg Config) (*App, error) {
	cfg = NormalizeConfig(cfg)

	var err error
	cfg, err = PrepareInstanceConfig(cfg)
	if err != nil {
		return nil, err
	}

	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}

	nodeID, err := LoadOrCreateNodeID(cfg.DataDir)
	if err != nil {
		return nil, err
	}

	app := newApp(cfg, nodeID)
	if cfg.ModelPath != "" {
		app.modelArtifactSHA256, err = computeModelArtifactSHA256(cfg.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("compute configured model identity: %w", err)
		}
		// newApp publishes an initial local member. Replace it now that the
		// artifact identity is available.
		app.store.Upsert(app.selfMember(time.Now().UTC()))
	}
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

func (a *App) configureHTTPServer() error {
	coordinatorRouter, err := a.coordinatorRouter()
	if err != nil {
		return err
	}
	stageRunner, err := a.stageRunner()
	if err != nil {
		return err
	}
	runtimeDeployment, err := a.runtimeDeploymentGateway()
	if err != nil {
		return err
	}
	publicRouter, err := a.publicRouter(coordinatorRouter)
	if err != nil {
		return err
	}
	a.server = a.httpServer(publicRouter, stageRunner, runtimeDeployment)
	return nil
}

func (a *App) coordinatorRouter() (http.Handler, error) {
	registry, err := modelregistry.Load(a.cfg.ModelsPath)
	if err != nil {
		return nil, fmt.Errorf("load model registry: %w", err)
	}
	server := coordinator.NewServer(
		registry,
		coordinator.WithNodeID(a.nodeID),
		coordinator.WithClusterToken(a.cfg.ClusterToken),
		coordinator.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(a.cfg.BenchmarksPath)),
		coordinator.WithMembershipSource(a.store, a.cfg.StaleAfter),
	)
	return server.Router(), nil
}

func (a *App) stageRunner() (http.Handler, error) {
	return runtimebridge.NewStageProxy(a.cfg.RuntimeURL)
}

func (a *App) runtimeDeploymentGateway() (http.Handler, error) {
	return runtimebridge.NewDeploymentProxy(a.cfg.RuntimeURL)
}

func (a *App) publicRouter(coordinatorRouter http.Handler) (http.Handler, error) {
	// All public APIs, including /v1/chat/completions, are coordinator-owned.
	// Followers proxy them to the elected leader through the facade router.
	return coordinatorRouter, nil
}

func (a *App) configureDiscovery() {
	self := func() membership.Member { return a.selfMember(time.Now().UTC()) }
	a.discovery = a.discoverySource(self)
	a.mdnsAdvertiser = a.newMDNSAdvertiser(self)
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

func (a *App) httpServer(coordinatorRouter http.Handler, stageRunner http.Handler, runtimeDeployment http.Handler) *http.Server {
	return &http.Server{
		Addr: a.cfg.Listen,
		Handler: facade.NewRouter(facade.Config{
			SelfID:            a.nodeID,
			ClusterToken:      a.cfg.ClusterToken,
			Store:             a.store,
			StaleAfter:        a.cfg.StaleAfter,
			Coordinator:       coordinatorRouter,
			StageRunner:       stageRunner,
			RuntimeDeployment: runtimeDeployment,
		}),
		ReadHeaderTimeout: 5 * time.Second,
	}
}

func (a *App) Run(ctx context.Context) error {
	listener, err := net.Listen("tcp", a.cfg.Listen)
	if err != nil {
		return fmt.Errorf("bind node API %s: %w", a.cfg.Listen, err)
	}

	a.cfg = a.cfg.WithBoundAPIURL(listener)

	runtimeSupervisor, runtimeURL, err := StartRuntimeSupervisor(ctx, a.cfg)
	if err != nil {
		_ = listener.Close()
		return err
	}
	a.runtimeSupervisor = runtimeSupervisor
	a.cfg.RuntimeURL = runtimeURL

	if err := a.configureHTTPServer(); err != nil {
		_ = listener.Close()
		return err
	}

	a.store.Upsert(a.selfMember(time.Now().UTC()))
	a.configureDiscovery()
	a.startMDNS(ctx)
	go a.discoveryLoop(ctx)

	errCh := make(chan error, 1)
	go func() {
		log.Printf(
			"JetsonFabric node listening on %s advertised=%s cluster=%s node_id=%s node_name=%s runtime=%s role=%s discovery=%v",
			listener.Addr(),
			a.cfg.APIURL,
			a.cfg.ClusterID,
			a.nodeID,
			a.cfg.NodeName,
			a.cfg.RuntimeURL,
			a.cfg.Role,
			a.cfg.DiscoveryModes,
		)
		errCh <- a.server.Serve(listener)
	}()

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

func (a *App) shutdown(errCh <-chan error) error {
	if a.mdnsAdvertiser != nil {
		a.mdnsAdvertiser.Shutdown()
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if a.server != nil {
		if err := a.server.Shutdown(shutdownCtx); err != nil {
			return err
		}
	}

	if a.runtimeSupervisor != nil {
		if err := a.runtimeSupervisor.Stop(shutdownCtx); err != nil {
			log.Printf("runtime supervisor shutdown: %v", err)
		}
	}

	if errCh == nil {
		return nil
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
		ClusterID:        a.cfg.ClusterID,
		NodeID:           a.nodeID,
		NodeName:         nodeName,
		Hostname:         snapshot.Hostname,
		Role:             a.cfg.Role,
		APIURL:           a.cfg.APIURL,
		RuntimeURL:       a.cfg.RuntimeURL,
		LeaderPreference: a.cfg.LeaderPreference,
		Arch:             snapshot.Arch,
		OS:               snapshot.OS,
		Capabilities:     a.memberCapabilities(snapshot.Capabilities),
		Metrics:          snapshot.Metrics,
		Engines:          a.engineEndpoints(),
		StartedAt:        a.startedAt,
		LastSeen:         now,
	}
}
