package coordinator

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

const defaultDeploymentContextSize = 4096

func (c *DeploymentController) loadPlan(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	members []membership.Member,
	spec deploymentSpec,
) error {
	byNode := make(map[string]membership.Member, len(members))
	for _, member := range members {
		byNode[member.NodeID] = member
	}
	for _, stage := range plan.Stages() {
		member, ok := byNode[stage.NodeID]
		if !ok {
			return fmt.Errorf("deployment member %q disappeared from the immutable snapshot", stage.NodeID)
		}
		if err := c.loadStage(ctx, plan, model, member, stage, spec); err != nil {
			return err
		}
	}
	return nil
}

func (c *DeploymentController) loadStage(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	member membership.Member,
	stage clusterplan.Stage,
	spec deploymentSpec,
) error {
	loadRequest := newRuntimeLoadRequest(plan, model, member, stage, spec)
	response, err := c.runtimeClient.Load(ctx, stage.APIURL, loadRequest)
	if err != nil {
		return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
	}
	if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "ready", false); err != nil {
		return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
	}
	return nil
}

func newRuntimeLoadRequest(
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	member membership.Member,
	stage clusterplan.Stage,
	spec deploymentSpec,
) runtimebridge.LoadDeploymentRequest {
	ctxSize := spec.ContextSize
	if ctxSize == 0 {
		ctxSize = defaultDeploymentContextSize
	}
	backend := cluster.ComputeBackend(capabilityString(member.Capabilities, cluster.CapabilityRuntimeComputeBackend))
	nGPULayers := 0
	if backend == cluster.ComputeBackendCUDA {
		nGPULayers = 999
	}
	if spec.NGPULayers != nil {
		nGPULayers = *spec.NGPULayers
	}
	return runtimebridge.LoadDeploymentRequest{
		DeploymentID: plan.Identity().DeploymentID, Epoch: plan.Identity().Epoch,
		ModelID: model.ID, ModelSHA256: plan.Model().ModelSHA256,
		Engine: string(plan.Model().Engine), ComputeBackend: string(backend),
		ModelPath: model.ArtifactPath, CtxSize: ctxSize, NGPULayers: nGPULayers,
		Threads: spec.Threads, Mode: string(plan.Model().ExecutionMode),
		StageIndex: stage.StageIndex, StageCount: stage.StageCount,
		LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
	}
}

func (c *DeploymentController) activatePlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	for _, stage := range plan.Stages() {
		if err := c.activateStage(ctx, plan, stage); err != nil {
			return err
		}
	}
	return nil
}

func (c *DeploymentController) activateStage(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	stage clusterplan.Stage,
) error {
	response, err := c.runtimeClient.Activate(ctx, stage.APIURL, runtimeDeploymentIdentity(plan))
	if err != nil {
		return fmt.Errorf("activate deployment on node %q: %w", stage.NodeID, err)
	}
	if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "active", true); err != nil {
		return fmt.Errorf("activate deployment on node %q: %w", stage.NodeID, err)
	}
	return nil
}

func (c *DeploymentController) repairActivePlan(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	members []membership.Member,
	spec deploymentSpec,
) error {
	byNode := make(map[string]membership.Member, len(members))
	for _, member := range members {
		byNode[member.NodeID] = member
	}
	for _, stage := range plan.Stages() {
		member, ok := byNode[stage.NodeID]
		if !ok {
			return fmt.Errorf("deployment member %q disappeared from the immutable snapshot", stage.NodeID)
		}
		status, err := c.runtimeClient.Status(ctx, stage.APIURL)
		if err != nil {
			return fmt.Errorf("inspect deployment on node %q: %w", stage.NodeID, err)
		}
		if validateRuntimeStatus(status, plan, stage, "active", true) == nil {
			continue
		}
		if status.Resident {
			if runtimeDeploymentIdentityMatches(status.Deployment, runtimeDeploymentIdentity(plan)) &&
				validateRuntimeStatus(status, plan, stage, "ready", false) == nil {
				if err := c.activateStage(ctx, plan, stage); err != nil {
					return err
				}
				continue
			}
			return fmt.Errorf(
				"node %q has resident state %q that cannot be repaired in place",
				stage.NodeID,
				status.State,
			)
		}
		if err := c.loadStage(ctx, plan, model, member, stage, spec); err != nil {
			return err
		}
		if err := c.activateStage(ctx, plan, stage); err != nil {
			return err
		}
	}
	return nil
}

func (c *DeploymentController) drainPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	identity := runtimeDeploymentIdentity(plan)
	var failures []string
	for _, stage := range plan.Stages() {
		response, err := c.runtimeClient.Drain(ctx, stage.APIURL, identity)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q: %v", stage.NodeID, err))
			continue
		}
		if !runtimeDeploymentIdentityMatches(response.Deployment, identity) {
			failures = append(failures, fmt.Sprintf("node %q acknowledged a different deployment identity", stage.NodeID))
		}
	}
	return joinedFailures(failures)
}

func (c *DeploymentController) unloadPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	identity := runtimeDeploymentIdentity(plan)
	var failures []string
	for _, stage := range plan.Stages() {
		response, err := c.runtimeClient.Unload(ctx, stage.APIURL, identity)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q: %v", stage.NodeID, err))
			continue
		}
		if !runtimeDeploymentIdentityMatches(response.Deployment, identity) {
			failures = append(failures, fmt.Sprintf("node %q acknowledged a different deployment identity", stage.NodeID))
			continue
		}
		if response.Resident || response.Active || response.State != "idle" {
			failures = append(failures, fmt.Sprintf("node %q did not release the deployment", stage.NodeID))
		}
	}
	return joinedFailures(failures)
}

func joinedFailures(failures []string) error {
	if len(failures) == 0 {
		return nil
	}
	sort.Strings(failures)
	return errors.New(strings.Join(failures, "; "))
}
