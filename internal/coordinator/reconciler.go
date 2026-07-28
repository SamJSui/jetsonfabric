package coordinator

import (
	"context"
	"errors"
	"log"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

// NotifyMembershipChanged coalesces membership refreshes. Reconciliation also
// retries periodically so a transient runtime or network failure can recover
// without another topology change.
func (s *Server) NotifyMembershipChanged() {
	s.deployments.notifyMembershipChanged()
}

func (c *DeploymentController) notifyMembershipChanged() {
	select {
	case c.reconcileCh <- struct{}{}:
	default:
	}
}

func (s *Server) RunReconciler(ctx context.Context) {
	s.deployments.runReconciler(ctx)
}

func (c *DeploymentController) runReconciler(ctx context.Context) {
	ticker := time.NewTicker(c.reconcileInterval)
	defer ticker.Stop()
	c.notifyMembershipChanged()
	for {
		select {
		case <-ctx.Done():
			return
		case <-c.reconcileCh:
			c.runReconcileAttempt(ctx)
		case <-ticker.C:
			c.runReconcileAttempt(ctx)
		}
	}
}

func (c *DeploymentController) runReconcileAttempt(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, c.transitionTimeout)
	defer cancel()
	if err := c.reconcile(ctx); err != nil && !errors.Is(err, context.Canceled) {
		log.Printf("deployment reconciliation pending: %v", err)
	}
}

// Reconcile computes a desired epoch from the last successful deployment
// intent and the current membership snapshot.
func (s *Server) Reconcile(ctx context.Context) error {
	return s.deployments.reconcile(ctx)
}

func (c *DeploymentController) reconcile(ctx context.Context) error {
	c.reconcileMu.Lock()
	defer c.reconcileMu.Unlock()
	if c.isLeader != nil && !c.isLeader(c.now()) {
		return nil
	}
	pendingCleanupErr := c.retryDraining(ctx)
	intent, ok := c.state.activeIntent()
	if !ok {
		return pendingCleanupErr
	}
	_, cleanupErr, err := c.switchDeploymentLocked(ctx, intent.spec(), false)
	if err == nil {
		result := errors.Join(pendingCleanupErr, cleanupErr)
		c.state.recordReconcileError(result, true)
		return result
	}
	snapshot := c.state.snapshot()
	healthy := snapshot.Active != nil && activePlanHealthy(
		*snapshot.Active,
		c.members(),
		c.now(),
		c.memberStaleAfter,
	) && c.activeRuntimePlanHealthy(ctx, *snapshot.Active)
	result := errors.Join(pendingCleanupErr, err)
	c.state.recordReconcileError(result, healthy)
	return result
}

func (c *DeploymentController) activeRuntimePlanHealthy(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
) bool {
	for _, stage := range plan.Stages() {
		status, err := c.runtimeClient.Status(ctx, stage.APIURL)
		if err != nil || validateRuntimeStatus(status, plan, stage, "active", true) != nil {
			return false
		}
	}
	return true
}

func (c *DeploymentController) retryDraining(ctx context.Context) error {
	snapshot := c.state.snapshot()
	var failures []error
	for _, plan := range snapshot.Draining {
		if err := c.retirePlan(ctx, plan); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}
