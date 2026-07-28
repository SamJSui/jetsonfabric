package coordinator

import (
	"context"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

const (
	deploymentSwitchTimeout  = 30 * time.Minute
	deploymentCleanupTimeout = 10 * time.Minute
)

type deploymentControllerConfig struct {
	registry          modelregistry.Registry
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	planPolicy        clusterplan.Policy
	now               func() time.Time
	runtimeClient     runtimebridge.DeploymentClient
	transitionTimeout time.Duration
	cleanupTimeout    time.Duration
	reconcileInterval time.Duration
	isLeader          func(time.Time) bool
}

// deploymentSpec is the controller input after HTTP decoding and validation.
type deploymentSpec struct {
	DeploymentID         string
	ModelID              string
	StageCount           int
	AllowColocatedStages bool
	ContextSize          int
	Threads              int
	NGPULayers           *int
}

// DeploymentController owns desired deployment intent, epoch transitions, and
// reconciliation. HTTP handlers translate requests and delegate here.
type DeploymentController struct {
	registry          modelregistry.Registry
	memberSource      MemberSource
	memberStaleAfter  time.Duration
	planPolicy        clusterplan.Policy
	now               func() time.Time
	state             *deploymentState
	runtimeClient     runtimebridge.DeploymentClient
	transitionTimeout time.Duration
	cleanupTimeout    time.Duration
	reconcileInterval time.Duration
	isLeader          func(time.Time) bool
	reconcileMu       sync.Mutex
	reconcileCh       chan struct{}
}

func newDeploymentController(cfg deploymentControllerConfig) *DeploymentController {
	return &DeploymentController{
		registry:          cfg.registry,
		memberSource:      cfg.memberSource,
		memberStaleAfter:  cfg.memberStaleAfter,
		planPolicy:        cfg.planPolicy,
		now:               cfg.now,
		state:             newDeploymentState(),
		runtimeClient:     cfg.runtimeClient,
		transitionTimeout: cfg.transitionTimeout,
		cleanupTimeout:    cfg.cleanupTimeout,
		reconcileInterval: cfg.reconcileInterval,
		isLeader:          cfg.isLeader,
		reconcileCh:       make(chan struct{}, 1),
	}
}

func (c *DeploymentController) snapshot() deploymentSnapshot {
	return c.state.snapshot()
}

func (c *DeploymentController) admit(modelID string) (deploymentAdmission, error) {
	return c.state.admit(modelID)
}

func (c *DeploymentController) switchDeployment(
	ctx context.Context,
	spec deploymentSpec,
	force bool,
) (deploymentBuild, error, error) {
	c.reconcileMu.Lock()
	defer c.reconcileMu.Unlock()
	return c.switchDeploymentLocked(ctx, spec, force)
}

func (c *DeploymentController) activeIntent() (deploymentIntent, bool) {
	return c.state.activeIntent()
}

func (c *DeploymentController) members() []membership.Member {
	if c.memberSource == nil {
		return nil
	}
	return c.memberSource.List()
}
