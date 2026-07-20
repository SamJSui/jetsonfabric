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
	if got := plan.Stages()[0].NodeID; got != "node-a" {
		t.Fatalf("plan changed after input mutation: %q", got)
	}

	stages := plan.Stages()
	stages[0].NodeID = "mutated-output"
	if got := plan.Stages()[0].NodeID; got != "node-a" {
		t.Fatalf("plan changed after accessor result mutation: %q", got)
	}
}

func TestNewDeploymentPlanAcceptsOneSelectedDataParallelReplica(t *testing.T) {
	spec := validDeploymentPlanSpec()
	spec.Model.ExecutionMode = cluster.ExecutionModeDataParallel
	spec.Stages = spec.Stages[:1]
	spec.Stages[0].StageCount = 1
	spec.Stages[0].LayerEnd = spec.Model.LayerCount

	plan, err := NewDeploymentPlan(spec)
	if err != nil {
		t.Fatalf("NewDeploymentPlan() error = %v", err)
	}
	if plan.StageCount() != 1 || plan.Stages()[0].LayerEnd != spec.Model.LayerCount {
		t.Fatalf("unexpected data-parallel plan: %+v", plan.Stages())
	}
}

func TestNewDeploymentPlanRejectsInvalidIdentityOrModel(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*DeploymentPlanSpec)
		want   string
	}{
		{
			name: "missing deployment id",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Identity.DeploymentID = ""
			},
			want: "deployment_id",
		},
		{
			name: "zero epoch",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Identity.Epoch = 0
			},
			want: "epoch",
		},
		{
			name: "missing model id",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.ModelID = ""
			},
			want: "model_id",
		},
		{
			name: "invalid artifact hash",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.ModelSHA256 = "not-a-sha256"
			},
			want: "model_sha256",
		},
		{
			name: "missing engine",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.Engine = ""
			},
			want: "engine",
		},
		{
			name: "invalid layer count",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.LayerCount = 0
			},
			want: "layer_count",
		},
		{
			name: "tensor parallel unsupported",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.ExecutionMode = cluster.ExecutionModeTensorParallel
			},
			want: "unsupported execution mode",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validDeploymentPlanSpec()
			test.mutate(&spec)
			_, err := NewDeploymentPlan(spec)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewDeploymentPlan() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestNewDeploymentPlanRejectsInvalidStages(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*DeploymentPlanSpec)
		want   string
	}{
		{
			name: "no stages",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages = nil
			},
			want: "at least one stage",
		},
		{
			name: "unordered stage index",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[0].StageIndex = 1
			},
			want: "has index",
		},
		{
			name: "inconsistent stage count",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].StageCount = 3
			},
			want: "stage_count",
		},
		{
			name: "missing node id",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[0].NodeID = ""
			},
			want: "node_id",
		},
		{
			name: "duplicate node",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].NodeID = spec.Stages[0].NodeID
			},
			want: "assigned more than once",
		},
		{
			name: "invalid node api url",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[0].APIURL = "127.0.0.1:19180"
			},
			want: "api_url is invalid",
		},
		{
			name: "duplicate node api url",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].APIURL = spec.Stages[0].APIURL
			},
			want: "api_url",
		},
		{
			name: "missing physical host",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[0].PhysicalHostID = ""
			},
			want: "physical_host_id",
		},
		{
			name: "layer gap",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].LayerStart = 15
			},
			want: "want 14",
		},
		{
			name: "layer overlap",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].LayerStart = 13
			},
			want: "want 14",
		},
		{
			name: "empty layer range",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[0].LayerEnd = 0
			},
			want: "empty or reversed",
		},
		{
			name: "incomplete layer coverage",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Stages[1].LayerEnd = 27
			},
			want: "want layer_count 28",
		},
		{
			name: "multiple selected data replicas",
			mutate: func(spec *DeploymentPlanSpec) {
				spec.Model.ExecutionMode = cluster.ExecutionModeDataParallel
			},
			want: "exactly one selected replica",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validDeploymentPlanSpec()
			test.mutate(&spec)
			_, err := NewDeploymentPlan(spec)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewDeploymentPlan() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func validDeploymentPlanSpec() DeploymentPlanSpec {
	return DeploymentPlanSpec{
		Identity: DeploymentIdentity{
			DeploymentID: "deployment-a",
			Epoch:        7,
		},
		Model: DeploymentModelIdentity{
			ModelID:       "model-a",
			ModelSHA256:   strings.Repeat("a", 64),
			Engine:        cluster.EngineLlamaCPP,
			ExecutionMode: cluster.ExecutionModePipelineParallel,
			LayerCount:    28,
		},
		Stages: []Stage{
			{
				StageIndex: 0, StageCount: 2,
				NodeID: "node-a", NodeName: "dopey", Hostname: "dopey",
				PhysicalHostID: "dopey", APIURL: "http://dopey.local:19180",
				LayerStart: 0, LayerEnd: 14,
			},
			{
				StageIndex: 1, StageCount: 2,
				NodeID: "node-b", NodeName: "sleepy", Hostname: "sleepy",
				PhysicalHostID: "sleepy", APIURL: "http://sleepy.local:19180",
				LayerStart: 14, LayerEnd: 28,
			},
		},
	}
}
