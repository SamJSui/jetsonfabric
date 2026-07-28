package coordinator

import (
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func plansEquivalent(left, right clusterplan.DeploymentPlan) bool {
	return left.Model() == right.Model() && slices.Equal(left.Stages(), right.Stages())
}

func activePlanHealthy(
	plan clusterplan.DeploymentPlan,
	members []membership.Member,
	now time.Time,
	staleAfter time.Duration,
) bool {
	fresh := make(map[string]membership.Member, len(members))
	for _, member := range members {
		member = membership.Normalize(member)
		if member.Valid() && !member.IsStale(now, staleAfter) {
			fresh[member.NodeID] = member
		}
	}
	for _, stage := range plan.Stages() {
		member, ok := fresh[stage.NodeID]
		if !ok || member.APIURL != stage.APIURL {
			return false
		}
	}
	return true
}

func validateRuntimeStatus(
	status runtimebridge.DeploymentStatus,
	plan clusterplan.DeploymentPlan,
	stage clusterplan.Stage,
	state string,
	active bool,
) error {
	if !status.Resident || status.Active != active || status.State != state || status.Deployment == nil {
		return fmt.Errorf("unexpected runtime status resident=%t active=%t state=%q", status.Resident, status.Active, status.State)
	}
	wantIdentity := runtimeDeploymentIdentity(plan)
	if !runtimeDeploymentIdentityMatches(status.Deployment, wantIdentity) {
		return fmt.Errorf(
			"runtime reports deployment %q epoch %d model %q sha256 %q, want deployment %q epoch %d model %q sha256 %q for stage %d",
			status.Deployment.DeploymentID, status.Deployment.Epoch,
			status.Deployment.ModelID, status.Deployment.ModelSHA256,
			wantIdentity.DeploymentID, wantIdentity.Epoch,
			wantIdentity.ModelID, wantIdentity.ModelSHA256, stage.StageIndex,
		)
	}
	if err := validateRuntimeModelMemory(status, plan, stage, active); err != nil {
		return fmt.Errorf("stage %d model residency: %w", stage.StageIndex, err)
	}
	return nil
}

func runtimeDeploymentIdentity(plan clusterplan.DeploymentPlan) runtimebridge.DeploymentIdentity {
	return runtimebridge.DeploymentIdentity{
		DeploymentID: plan.Identity().DeploymentID,
		Epoch:        plan.Identity().Epoch,
		ModelID:      plan.Model().ModelID,
		ModelSHA256:  plan.Model().ModelSHA256,
	}
}

func runtimeDeploymentIdentityMatches(actual *runtimebridge.DeploymentIdentity, expected runtimebridge.DeploymentIdentity) bool {
	return actual != nil && *actual == expected
}

func validateRuntimeModelMemory(
	status runtimebridge.DeploymentStatus,
	plan clusterplan.DeploymentPlan,
	stage clusterplan.Stage,
	active bool,
) error {
	memory := status.ModelMemory
	if memory == nil {
		return errors.New("runtime omitted model_memory")
	}
	model := plan.Model()
	if memory.LayerStart != stage.LayerStart || memory.LayerEnd != stage.LayerEnd {
		return fmt.Errorf("reported layers [%d,%d), want [%d,%d)", memory.LayerStart, memory.LayerEnd, stage.LayerStart, stage.LayerEnd)
	}
	if memory.LayerCount != model.LayerCount {
		return fmt.Errorf("reported layer_count %d, want %d", memory.LayerCount, model.LayerCount)
	}
	if memory.ResidentWeightBytes == 0 || memory.TotalWeightBytes == 0 || memory.ResidentTensorCount == 0 {
		return errors.New("runtime reported empty model residency")
	}
	partitioned := stage.LayerStart != 0 || stage.LayerEnd != model.LayerCount
	if memory.Partitioned != partitioned {
		return fmt.Errorf("reported partitioned=%t, want %t", memory.Partitioned, partitioned)
	}
	if !partitioned && memory.ResidentWeightBytes != memory.TotalWeightBytes {
		return fmt.Errorf("full-range stage retains %d bytes from a %d-byte model", memory.ResidentWeightBytes, memory.TotalWeightBytes)
	}
	if partitioned && memory.ResidentWeightBytes >= memory.TotalWeightBytes {
		return fmt.Errorf("partition retains %d bytes from a %d-byte model", memory.ResidentWeightBytes, memory.TotalWeightBytes)
	}
	if memory.Pinned != active {
		return fmt.Errorf("reported pinned=%t, want %t", memory.Pinned, active)
	}
	return nil
}
