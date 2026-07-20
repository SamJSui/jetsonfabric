package clusterplan

import (
	"encoding/hex"
	"fmt"
	"net/url"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

// DeploymentIdentity distinguishes one cluster assignment from the model it
// serves. Epochs are monotonically assigned by the coordinator in later work.
type DeploymentIdentity struct {
	DeploymentID string
	Epoch        uint64
}

// DeploymentModelIdentity captures the correctness-critical model facts shared
// by every stage in one deployment plan.
type DeploymentModelIdentity struct {
	ModelID       string
	ModelSHA256   string
	Engine        cluster.Engine
	ExecutionMode cluster.ExecutionMode
	LayerCount    int
}

// DeploymentPlanSpec is mutable construction input. NewDeploymentPlan validates
// and copies it into an immutable DeploymentPlan value.
type DeploymentPlanSpec struct {
	Identity DeploymentIdentity
	Model    DeploymentModelIdentity
	Stages   []Stage
}

// DeploymentPlan is an immutable snapshot of one coordinator-selected cluster
// assignment. Accessors return values or copies so callers cannot mutate the
// plan after construction.
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

	if err := validateDeploymentIdentity(identity); err != nil {
		return DeploymentPlan{}, err
	}
	if err := validateDeploymentModelIdentity(model); err != nil {
		return DeploymentPlan{}, err
	}
	if err := validateDeploymentStages(model, stages); err != nil {
		return DeploymentPlan{}, err
	}

	return DeploymentPlan{
		identity: identity,
		model:    model,
		stages:   stages,
	}, nil
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

func validateDeploymentIdentity(identity DeploymentIdentity) error {
	if identity.DeploymentID == "" {
		return fmt.Errorf("deployment_id is required")
	}
	if identity.Epoch == 0 {
		return fmt.Errorf("deployment epoch must be positive")
	}
	return nil
}

func validateDeploymentModelIdentity(model DeploymentModelIdentity) error {
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
	case cluster.ExecutionModeDataParallel, cluster.ExecutionModePipelineParallel:
		return nil
	default:
		return fmt.Errorf("unsupported execution mode %q", model.ExecutionMode)
	}
}

func validateDeploymentStages(model DeploymentModelIdentity, stages []Stage) error {
	if len(stages) == 0 {
		return fmt.Errorf("deployment requires at least one stage")
	}
	if model.ExecutionMode == cluster.ExecutionModeDataParallel && len(stages) != 1 {
		return fmt.Errorf("data_parallel deployment requires exactly one selected replica")
	}

	seenNodeIDs := make(map[string]struct{}, len(stages))
	seenAPIURLs := make(map[string]struct{}, len(stages))
	expectedLayerStart := 0

	for index, stage := range stages {
		if stage.StageIndex != index {
			return fmt.Errorf("stage %d has index %d", index, stage.StageIndex)
		}
		if stage.StageCount != len(stages) {
			return fmt.Errorf("stage %d has stage_count %d, want %d", index, stage.StageCount, len(stages))
		}

		nodeID := strings.TrimSpace(stage.NodeID)
		if nodeID == "" {
			return fmt.Errorf("stage %d node_id is required", index)
		}
		if _, exists := seenNodeIDs[nodeID]; exists {
			return fmt.Errorf("node_id %q is assigned more than once", nodeID)
		}
		seenNodeIDs[nodeID] = struct{}{}

		apiURL := strings.TrimSpace(stage.APIURL)
		if !validNodeAPIURL(apiURL) {
			return fmt.Errorf("stage %d api_url is invalid", index)
		}
		if _, exists := seenAPIURLs[apiURL]; exists {
			return fmt.Errorf("api_url %q is assigned more than once", apiURL)
		}
		seenAPIURLs[apiURL] = struct{}{}

		if strings.TrimSpace(stage.PhysicalHostID) == "" {
			return fmt.Errorf("stage %d physical_host_id is required", index)
		}
		if stage.LayerStart != expectedLayerStart {
			return fmt.Errorf("stage %d layer range starts at %d, want %d", index, stage.LayerStart, expectedLayerStart)
		}
		if stage.LayerEnd <= stage.LayerStart {
			return fmt.Errorf("stage %d layer range is empty or reversed", index)
		}
		expectedLayerStart = stage.LayerEnd
	}

	if expectedLayerStart != model.LayerCount {
		return fmt.Errorf("stage ranges end at %d, want layer_count %d", expectedLayerStart, model.LayerCount)
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

func validNodeAPIURL(value string) bool {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" {
		return false
	}
	return parsed.Scheme == "http" || parsed.Scheme == "https"
}
