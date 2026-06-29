package layersplit

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestPlanForCandidatesSplitsEvenly(t *testing.T) {
	plan := mustPlan(t, 28, candidates("orin-01", "orin-02"))

	assertStage(t, plan.Stages[0], "orin-01", StageRoleFirst, 0, 14)
	assertStage(t, plan.Stages[1], "orin-02", StageRoleLast, 14, 28)
}

func TestPlanForCandidatesSplitsUnevenLayerCounts(t *testing.T) {
	plan := mustPlan(t, 28, candidates("orin-01", "orin-02", "orin-03"))

	assertStage(t, plan.Stages[0], "orin-01", StageRoleFirst, 0, 10)
	assertStage(t, plan.Stages[1], "orin-02", StageRoleMiddle, 10, 19)
	assertStage(t, plan.Stages[2], "orin-03", StageRoleLast, 19, 28)
}

func TestPlanForCandidatesWorksForLargerModels(t *testing.T) {
	plan := mustPlan(t, 40, candidates("orin-01", "orin-02", "orin-03", "orin-04"))

	assertStage(t, plan.Stages[0], "orin-01", StageRoleFirst, 0, 10)
	assertStage(t, plan.Stages[1], "orin-02", StageRoleMiddle, 10, 20)
	assertStage(t, plan.Stages[2], "orin-03", StageRoleMiddle, 20, 30)
	assertStage(t, plan.Stages[3], "orin-04", StageRoleLast, 30, 40)
}

func TestPlanForCandidatesCapsStagesAtLayerCount(t *testing.T) {
	plan := mustPlan(t, 2, candidates("orin-01", "orin-02", "orin-03"))

	if len(plan.Stages) != 2 {
		t.Fatalf("expected two usable stages, got %d", len(plan.Stages))
	}
	assertStage(t, plan.Stages[0], "orin-01", StageRoleFirst, 0, 1)
	assertStage(t, plan.Stages[1], "orin-02", StageRoleLast, 1, 2)
}

func TestPlanForCandidatesUsesWeights(t *testing.T) {
	plan := mustPlan(t, 28, []NodeCandidate{
		{NodeName: "orin-01", Weight: 1},
		{NodeName: "orin-02", Weight: 2},
		{NodeName: "orin-03", Weight: 1},
	})

	assertStage(t, plan.Stages[0], "orin-01", StageRoleFirst, 0, 7)
	assertStage(t, plan.Stages[1], "orin-02", StageRoleMiddle, 7, 21)
	assertStage(t, plan.Stages[2], "orin-03", StageRoleLast, 21, 28)
}

func TestPlanForModelRejectsMissingLayerMetadata(t *testing.T) {
	_, err := PlanForModel(cluster.ModelProfile{ID: "qwen"}, candidates("orin-01", "orin-02"))
	if err == nil {
		t.Fatal("expected missing layer metadata to fail")
	}
}

func mustPlan(t *testing.T, layerCount int, candidates []NodeCandidate) Plan {
	t.Helper()
	plan, err := PlanForCandidates("qwen", layerCount, candidates)
	if err != nil {
		t.Fatalf("plan layer split: %v", err)
	}
	if plan.Mode != cluster.RouteModeLayerSplit {
		t.Fatalf("unexpected mode: %s", plan.Mode)
	}
	return plan
}

func candidates(nodeNames ...string) []NodeCandidate {
	output := make([]NodeCandidate, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		output = append(output, NodeCandidate{NodeName: nodeName, Weight: 1})
	}
	return output
}

func assertStage(t *testing.T, stage Stage, nodeName string, role StageRole, start int, end int) {
	t.Helper()
	if stage.NodeName != nodeName || stage.Role != role || stage.LayerStart != start || stage.LayerEnd != end {
		t.Fatalf("unexpected stage: %+v", stage)
	}
	if stage.LayerCount != end-start {
		t.Fatalf("unexpected layer count: %+v", stage)
	}
}
