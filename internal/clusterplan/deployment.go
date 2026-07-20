package clusterplan

import (
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

// DeploymentIdentity distinguishes one cluster assignment from the model it serves.
type DeploymentIdentity struct {
	DeploymentID string
	Epoch        uint64
}

// DeploymentModelIdentity records correctness-critical facts shared by all stages.
type DeploymentModelIdentity struct {
	ModelID       string
	ModelSHA256   string
	Engine        cluster.Engine
	ExecutionMode cluster.ExecutionMode
	LayerCount    int
}

// DeploymentPlanSpec is mutable constructor input.
type DeploymentPlanSpec struct {
	Identity DeploymentIdentity
	Model    DeploymentModelIdentity
	Stages   []Stage
}

// DeploymentPlan is immutable after construction. Slice accessors return copies.
type DeploymentPlan struct {
	identity DeploymentIdentity
	model    DeploymentModelIdentity
	stages   []Stage
}

func NewDeploymentPlan(spec DeploymentPlanSpec) (DeploymentPlan, error) {
	identity := DeploymentIdentity{
		DeploymentID: strings.TrimSpace(spec.Identity.DeploymentID),
		Epoch:        spec.Identity.Epoch,
	}
	model := DeploymentModelIdentity{
		ModelID:       strings.TrimSpace(spec.Model.ModelID),
		ModelSHA256:   strings.ToLower(strings.TrimSpace(spec.Model.ModelSHA256)),
		Engine:        cluster.Engine(strings.TrimSpace(string(spec.Model.Engine))),
		ExecutionMode: spec.Model.ExecutionMode,
		LayerCount:    spec.Model.LayerCount,
	}
	stages := append([]Stage(nil), spec.Stages...)

	if err := validateDeploymentPlan(identity, model, stages); err != nil {
		return DeploymentPlan{}, err
	}
	return DeploymentPlan{identity: identity, model: model, stages: stages}, nil
}

func (plan DeploymentPlan) Identity() DeploymentIdentity {
	return plan.identity
}

func (plan DeploymentPlan) Model() DeploymentModelIdentity {
	return plan.model
}

func (plan DeploymentPlan) Stages() []Stage {
	return append([]Stage(nil), plan.stages...)
}

func (plan DeploymentPlan) StageCount() int {
	return len(plan.stages)
}

func validateDeploymentPlan(
	identity DeploymentIdentity,
	model DeploymentModelIdentity,
	stages []Stage,
) error {
	if identity.DeploymentID == "" {
		return fmt.Errorf("deployment_id is required")
	}
	if identity.Epoch == 0 {
		return fmt.Errorf("deployment epoch must be positive")
	}
	if model.ModelID == "" {
		return fmt.Errorf("model_id is required")
	}
	if !validSHA256(model.ModelSHA256) {
		return fmt.Errorf("model_sha256 must be a 64-character hexadecimal digest")
	}
	if model.Engine == "" {
		return fmt.Errorf("engine is required")
	}
	if model.LayerCount <= 0 {
		return fmt.Errorf("layer_count must be positive")
	}
	switch model.ExecutionMode {
	case cluster.ExecutionModeDataParallel:
		if len(stages) != 1 {
			return fmt.Errorf("data_parallel deployment requires exactly one selected replica")
		}
	case cluster.ExecutionModePipelineParallel:
	default:
		return fmt.Errorf("unsupported execution mode %q", model.ExecutionMode)
	}
	if len(stages) == 0 {
		return fmt.Errorf("deployment requires at least one stage")
	}

	seenNodes := make(map[string]struct{}, len(stages))
	expectedStart := 0
	for index, stage := range stages {
		if stage.StageIndex != index || stage.StageCount != len(stages) {
			return fmt.Errorf("stage %d has inconsistent index or count", index)
		}
		nodeID := strings.TrimSpace(stage.NodeID)
		if nodeID == "" {
			return fmt.Errorf("stage %d node_id is required", index)
		}
		if _, exists := seenNodes[nodeID]; exists {
			return fmt.Errorf("node_id %q is assigned more than once", nodeID)
		}
		seenNodes[nodeID] = struct{}{}
		if stage.LayerStart != expectedStart {
			return fmt.Errorf("stage %d layer range starts at %d, want %d", index, stage.LayerStart, expectedStart)
		}
		if stage.LayerEnd <= stage.LayerStart {
			return fmt.Errorf("stage %d layer range is empty or reversed", index)
		}
		expectedStart = stage.LayerEnd
	}
	if expectedStart != model.LayerCount {
		return fmt.Errorf("stage ranges end at %d, want layer_count %d", expectedStart, model.LayerCount)
	}
	return nil
}

func validSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
