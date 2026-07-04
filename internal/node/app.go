package node

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/control"
	"github.com/SamJSui/jetsonfabric/internal/discovery"
	"github.com/SamJSui/jetsonfabric/internal/facade"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

type App struct {
	cfg       Config
	nodeID    string
	startedAt time.Time
	store     *membership.Store
	discovery discovery.Source
	server    *http.Server
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
	startedAt := time.Now().UTC()

	registry, err := modelregistry.Load(cfg.ModelsPath)
	if err != nil {
		return nil, fmt.Errorf("load model registry: %w", err)
	}
	coordinator := control.NewServer(
		cfg.JoinToken,
		registry,
		control.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(cfg.BenchmarksPath)),
	)

	store := membership.NewStore()
	app := &App{
		cfg:       cfg,
		nodeID:    nodeID,
		startedAt: startedAt,
		store:     store,
	}
	store.Upsert(app.selfMember(time.Now().UTC()))

	app.discovery = discovery.NewStaticSource(cfg.Seeds, func() membership.Member {
		return app.selfMember(time.Now().UTC())
	})
	app.server = &http.Server{
		Addr: cfg.Listen,
		Handler: facade.NewRouter(facade.Config{
			SelfID:      nodeID,
			Store:       store,
			StaleAfter:  cfg.StaleAfter,
			Coordinator: coordinator.Router(),
		}),
		ReadHeaderTimeout: 5 * time.Second,
	}

	return app, nil
}

func (a *App) Run(ctx context.Context) error {
	go a.discoveryLoop(ctx)

	errCh := make(chan error, 1)
	go func() {
		log.Printf("JetsonFabric node listening on http://%s cluster=%s node_id=%s", a.cfg.Listen, a.cfg.ClusterID, a.nodeID)
		errCh <- a.server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := a.server.Shutdown(shutdownCtx); err != nil {
			return err
		}
		if err := <-errCh; err != nil && err != http.ErrServerClosed {
			return err
		}
		return nil
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
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

	members, err := a.discovery.Discover(ctx)
	if err != nil {
		log.Printf("discovery failed: %v", err)
		return
	}
	for _, member := range members {
		if member.ClusterID != a.cfg.ClusterID {
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
