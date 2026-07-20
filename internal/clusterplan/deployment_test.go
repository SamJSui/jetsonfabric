package clusterplan

import (
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestNewDeploymentPlanCopiesAndNormalizesInput(t *testing.T) {
	spec := validDeploymentPlanSpec()
	spec.Identity.DeploymentID = "  deployment-a  "
	spec.Model.ModelID = "  model-a  "
	spec.Model.ModelSHA256 = strings.Repeat("A", 64)

	plan, err := NewDeploymentPlan(spec)
	if err != nil {
		t.Fatalf("NewDeploymentPlan() error = %v", err)
	}
	if plan.Identity() != (DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 7}) {
		t.Fatalf("unexpected deployment identity: %+v", plan.Identity())
	}
	if plan.Model().ModelID != "model-a" || plan.Model().ModelSHA256 != strings.Repeat("a", 64) {
		t.Fatalf("unexpected model identity: %+v", plan.Model())
	}
	if plan.StageCount() != 2 {
		t.Fatalf("StageCount() = %d, want 2", plan.StageCount())
	}

	spec.Stages[0].NodeID = "mutated-input"
	if plan.Stages()[0].NodeID != "node-a" {
		t.Fatal("plan changed after constructor input mutation")
	}
	stages := plan.Stages()
	stages[0].NodeID = "mutated-output"
	if plan.Stages()[0].NodeID != "node-a" {
		t.Fatal("plan changed after accessor result mutation")
	}
}

func TestNewDeploymentPlanAcceptsOneDataParallelReplica(t *testing.T) {
	spec := validDeploymentPlanSpec()
	spec.Model.ExecutionMode = cluster.ExecutionModeDataParallel
	spec.Stages = spec.Stages[:1]
	spec.Stages[0].StageCount = 1
	spec.Stages[0].LayerEnd = spec.Model.LayerCount

	plan, err := NewDeploymentPlan(spec)
	if err != nil {
		t.Fatalf("NewDeploymentPlan() error = %v", err)
	}
	if plan.StageCount() != 1 {
		t.Fatalf("StageCount() = %d, want 1", plan.StageCount())
	}
}

func TestNewDeploymentPlanRejectsInvalidMetadata(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*DeploymentPlanSpec)
		want   string
	}{
		{"missing deployment id", func(s *DeploymentPlanSpec) { s.Identity.DeploymentID = "" }, "deployment_id"},
		{"zero epoch", func(s *DeploymentPlanSpec) { s.Identity.Epoch = 0 }, "epoch"},
		{"missing model id", func(s *DeploymentPlanSpec) { s.Model.ModelID = "" }, "model_id"},
		{"invalid artifact hash", func(s *DeploymentPlanSpec) { s.Model.ModelSHA256 = "bad" }, "model_sha256"},
		{"missing engine", func(s *DeploymentPlanSpec) { s.Model.Engine = "" }, "engine"},
		{"invalid layer count", func(s *DeploymentPlanSpec) { s.Model.LayerCount = 0 }, "layer_count"},
		{"tensor parallel", func(s *DeploymentPlanSpec) { s.Model.ExecutionMode = cluster.ExecutionModeTensorParallel }, "unsupported execution mode"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validDeploymentPlanSpec()
			test.mutate(&spec)
			expectDeploymentPlanError(t, spec, test.want)
		})
	}
}

func TestNewDeploymentPlanRejectsInvalidStages(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*DeploymentPlanSpec)
		want   string
	}{
		{"no stages", func(s *DeploymentPlanSpec) { s.Stages = nil }, "at least one stage"},
		{"wrong stage index", func(s *DeploymentPlanSpec) { s.Stages[0].StageIndex = 1 }, "inconsistent"},
		{"wrong stage count", func(s *DeploymentPlanSpec) { s.Stages[1].StageCount = 3 }, "inconsistent"},
		{"missing node", func(s *DeploymentPlanSpec) { s.Stages[0].NodeID = "" }, "node_id"},
		{"duplicate node", func(s *DeploymentPlanSpec) { s.Stages[1].NodeID = s.Stages[0].NodeID }, "assigned more than once"},
		{"layer gap", func(s *DeploymentPlanSpec) { s.Stages[1].LayerStart = 15 }, "want 14"},
		{"empty range", func(s *DeploymentPlanSpec) { s.Stages[0].LayerEnd = 0 }, "empty or reversed"},
		{"incomplete coverage", func(s *DeploymentPlanSpec) { s.Stages[1].LayerEnd = 27 }, "layer_count 28"},
		{"multiple data replicas", func(s *DeploymentPlanSpec) { s.Model.ExecutionMode = cluster.ExecutionModeDataParallel }, "exactly one selected replica"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validDeploymentPlanSpec()
			test.mutate(&spec)
			expectDeploymentPlanError(t, spec, test.want)
		})
	}
}

func expectDeploymentPlanError(t *testing.T, spec DeploymentPlanSpec, want string) {
	t.Helper()
	_, err := NewDeploymentPlan(spec)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("NewDeploymentPlan() error = %v, want containing %q", err, want)
	}
}

func validDeploymentPlanSpec() DeploymentPlanSpec {
	return DeploymentPlanSpec{
		Identity: DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 7},
		Model: DeploymentModelIdentity{
			ModelID: "model-a", ModelSHA256: strings.Repeat("a", 64),
			Engine: cluster.EngineLlamaCPP, ExecutionMode: cluster.ExecutionModePipelineParallel,
			LayerCount: 28,
		},
		Stages: []Stage{
			{
				StageIndex: 0, StageCount: 2, NodeID: "node-a", NodeName: "dopey",
				PhysicalHostID: "dopey", APIURL: "http://dopey.local:19180",
				LayerStart: 0, LayerEnd: 14,
			},
			{
				StageIndex: 1, StageCount: 2, NodeID: "node-b", NodeName: "sleepy",
				PhysicalHostID: "sleepy", APIURL: "http://sleepy.local:19180",
				LayerStart: 14, LayerEnd: 28,
			},
		},
	}
}
