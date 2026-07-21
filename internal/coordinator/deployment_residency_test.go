package coordinator

import (
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func TestValidateRuntimeStatusRequiresAssignedPartitionResidency(t *testing.T) {
	plan, err := clusterplan.NewDeploymentPlan(clusterplan.DeploymentPlanSpec{
		Identity: clusterplan.DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 1},
		Model: clusterplan.DeploymentModelIdentity{
			ModelID:       "model-a",
			ModelSHA256:   strings.Repeat("a", 64),
			Engine:        cluster.EngineLlamaCPP,
			ExecutionMode: cluster.ExecutionModePipelineParallel,
			LayerCount:    4,
		},
		Stages: []clusterplan.Stage{
			{StageIndex: 0, StageCount: 2, NodeID: "node-a", LayerStart: 0, LayerEnd: 2},
			{StageIndex: 1, StageCount: 2, NodeID: "node-b", LayerStart: 2, LayerEnd: 4},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	stage := plan.Stages()[0]
	valid := runtimebridge.DeploymentStatus{
		Resident: true,
		State:    "ready",
		Deployment: &runtimebridge.DeploymentIdentity{
			DeploymentID: "deployment-a",
			ModelID:      "model-a",
		},
		ModelMemory: &runtimebridge.ModelMemory{
			LayerStart:          0,
			LayerEnd:            2,
			LayerCount:          4,
			ResidentWeightBytes: 180,
			TotalWeightBytes:    400,
			ResidentTensorCount: 12,
			Partitioned:         true,
		},
	}
	if err := validateRuntimeStatus(valid, plan, stage, "ready", false); err != nil {
		t.Fatalf("valid partition status rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*runtimebridge.DeploymentStatus)
	}{
		{name: "missing accounting", mutate: func(status *runtimebridge.DeploymentStatus) { status.ModelMemory = nil }},
		{name: "wrong range", mutate: func(status *runtimebridge.DeploymentStatus) { status.ModelMemory.LayerEnd = 3 }},
		{name: "full model retained", mutate: func(status *runtimebridge.DeploymentStatus) {
			status.ModelMemory.ResidentWeightBytes = status.ModelMemory.TotalWeightBytes
		}},
		{name: "ready partition pinned", mutate: func(status *runtimebridge.DeploymentStatus) { status.ModelMemory.Pinned = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			status := valid
			memory := *valid.ModelMemory
			status.ModelMemory = &memory
			test.mutate(&status)
			if err := validateRuntimeStatus(status, plan, stage, "ready", false); err == nil {
				t.Fatal("invalid model residency was accepted")
			}
		})
	}
}

func TestValidateRuntimeStatusRequiresCompleteFullModelResidency(t *testing.T) {
	plan, err := clusterplan.NewDeploymentPlan(clusterplan.DeploymentPlanSpec{
		Identity: clusterplan.DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 1},
		Model: clusterplan.DeploymentModelIdentity{
			ModelID:       "model-a",
			ModelSHA256:   strings.Repeat("a", 64),
			Engine:        cluster.EngineLlamaCPP,
			ExecutionMode: cluster.ExecutionModePipelineParallel,
			LayerCount:    4,
		},
		Stages: []clusterplan.Stage{
			{StageIndex: 0, StageCount: 1, NodeID: "node-a", LayerStart: 0, LayerEnd: 4},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	status := runtimebridge.DeploymentStatus{
		Resident: true,
		State:    "ready",
		Deployment: &runtimebridge.DeploymentIdentity{
			DeploymentID: "deployment-a",
			ModelID:      "model-a",
		},
		ModelMemory: &runtimebridge.ModelMemory{
			LayerEnd:            4,
			LayerCount:          4,
			ResidentWeightBytes: 400,
			TotalWeightBytes:    400,
			ResidentTensorCount: 24,
		},
	}
	stage := plan.Stages()[0]
	if err := validateRuntimeStatus(status, plan, stage, "ready", false); err != nil {
		t.Fatalf("valid full-model status rejected: %v", err)
	}
	status.ModelMemory.ResidentWeightBytes--
	if err := validateRuntimeStatus(status, plan, stage, "ready", false); err == nil {
		t.Fatal("incomplete full-model residency was accepted")
	}
}
