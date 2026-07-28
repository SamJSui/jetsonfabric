package coordinator

import (
	"context"
	"errors"
	"fmt"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type deploymentBuild struct {
	model   cluster.ModelProfile
	members []membership.Member
	policy  clusterplan.Policy
	result  clusterplan.DeploymentBuildResult
}

func (c *DeploymentController) switchDeploymentLocked(
	ctx context.Context,
	spec deploymentSpec,
	force bool,
) (deploymentBuild, error, error) {
	build, err := c.buildDeployment(spec)
	if err != nil {
		return deploymentBuild{}, nil, fmt.Errorf("%w: %v", errDeploymentPlanInvalid, err)
	}
	current := c.state.snapshot()
	if !force && current.Active != nil && plansEquivalent(*current.Active, build.result.Plan) {
		return build, nil, nil
	}
	cleanupErr, err := c.transitionDeployment(ctx, build, spec)
	return build, cleanupErr, err
}

func (c *DeploymentController) buildDeployment(spec deploymentSpec) (deploymentBuild, error) {
	model, ok := c.registry.Find(spec.ModelID)
	if !ok {
		return deploymentBuild{}, fmt.Errorf("model %q is not in the registry", spec.ModelID)
	}
	if c.memberSource == nil {
		return deploymentBuild{}, errDeploymentUnavailable
	}
	snapshot := c.state.snapshot()
	identity := clusterplan.DeploymentIdentity{
		DeploymentID: spec.DeploymentID,
		Epoch:        snapshot.ProposedEpoch,
	}
	if identity.DeploymentID == "" {
		identity.DeploymentID = fmt.Sprintf("deployment-%d-%d", identity.Epoch, c.now().UnixNano())
	}
	policy := deploymentPolicy(c.planPolicy, spec)
	members := append([]membership.Member(nil), c.memberSource.List()...)
	result, err := clusterplan.BuildDeploymentPlan(clusterplan.DeploymentBuildRequest{
		Identity: identity, Model: model, Members: members,
		Now: c.now(), StaleAfter: c.memberStaleAfter, Policy: policy,
	})
	if err != nil {
		return deploymentBuild{}, err
	}
	return deploymentBuild{model: model, members: members, policy: policy, result: result}, nil
}

func deploymentPolicy(base clusterplan.Policy, spec deploymentSpec) clusterplan.Policy {
	policy := base
	if spec.StageCount > 0 {
		policy.StageCount = spec.StageCount
	}
	if spec.AllowColocatedStages {
		policy.AllowColocatedStages = true
	}
	return policy
}

func (c *DeploymentController) transitionDeployment(
	ctx context.Context,
	build deploymentBuild,
	spec deploymentSpec,
) (error, error) {
	previous, err := c.state.beginTransition(build.result.Plan)
	if err != nil {
		return nil, err
	}
	if err := c.preparePlan(ctx, build, spec); err != nil {
		failure := c.rollbackPreparedPlan(build.result.Plan, previous, err)
		return nil, failure
	}

	intent := intentFromSpec(spec, build.policy)
	previous = c.state.publish(build.result.Plan, intent)
	if previous == nil {
		return nil, nil
	}
	if err := c.retirePlan(ctx, *previous); err != nil {
		c.state.recordReconcileError(err, true)
		return err, nil
	}
	return nil, nil
}

func (c *DeploymentController) preparePlan(
	ctx context.Context,
	build deploymentBuild,
	spec deploymentSpec,
) error {
	if err := c.loadPlan(ctx, build.result.Plan, build.model, build.members, spec); err != nil {
		return fmt.Errorf("prepare deployment: %w", err)
	}
	if err := c.activatePlan(ctx, build.result.Plan); err != nil {
		return fmt.Errorf("activate deployment: %w", err)
	}
	return nil
}

func (c *DeploymentController) rollbackPreparedPlan(
	plan clusterplan.DeploymentPlan,
	previous *clusterplan.DeploymentPlan,
	cause error,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), c.cleanupTimeout)
	defer cancel()
	cleanupErr := c.cleanupPlan(ctx, plan)
	healthy := previous == nil || activePlanHealthy(*previous, c.members(), c.now(), c.memberStaleAfter)
	if cleanupErr != nil {
		cause = fmt.Errorf("%w; rollback cleanup: %v", cause, cleanupErr)
	}
	c.state.rollback(cause, healthy)
	return cause
}

func (c *DeploymentController) retirePlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	drainErr := c.drainPlan(ctx, plan)
	if err := c.state.waitForEpoch(ctx, plan.Identity().Epoch); err != nil {
		return errors.Join(
			drainErr,
			fmt.Errorf("wait for deployment %q sessions: %w", plan.Identity().DeploymentID, err),
		)
	}
	unloadErr := c.unloadPlan(ctx, plan)
	if drainErr != nil || unloadErr != nil {
		return errors.Join(
			wrappedPlanError("drain", plan, drainErr),
			wrappedPlanError("unload", plan, unloadErr),
		)
	}
	c.state.finishDraining(plan.Identity().Epoch)
	return nil
}

func wrappedPlanError(operation string, plan clusterplan.DeploymentPlan, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s deployment %q: %w", operation, plan.Identity().DeploymentID, err)
}

func (c *DeploymentController) cleanupPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	drainErr := c.drainPlan(ctx, plan)
	unloadErr := c.unloadPlan(ctx, plan)
	switch {
	case drainErr != nil && unloadErr != nil:
		return fmt.Errorf("drain: %v; unload: %v", drainErr, unloadErr)
	case drainErr != nil:
		return drainErr
	default:
		return unloadErr
	}
}
