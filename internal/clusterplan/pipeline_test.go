package clusterplan

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestPreviewPipelinePromotesOneStageFullModelRoute(t *testing.T) {
	member := clusterPlanTestMember("node-a", "dopey", "dopey", "http://dopey.local:52415")
	advertiseRuntimeAssignment(&member, 0, 1, 0, 28)

	preview := PreviewPipeline(Request{
		Model:      clusterPlanTestModel(),
		Members:    []membership.Member{member},
		Now:        clusterPlanTestNow(),
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 1},
	})
	if !preview.Valid || preview.Mode != cluster.ExecutionModePipelineParallel {
		t.Fatalf("expected valid one-stage pipeline: %+v", preview)
	}
	if preview.StageCount != 1 || len(preview.Stages) != 1 {
		t.Fatalf("unexpected stage count: %+v", preview)
	}
	stage := preview.Stages[0]
	if stage.StageIndex != 0 || stage.StageCount != 1 || stage.LayerStart != 0 || stage.LayerEnd != 28 {
		t.Fatalf("unexpected one-stage assignment: %+v", stage)
	}
}

func TestPreviewPipelinePreservesMultiStagePipeline(t *testing.T) {
	preview := PreviewPipeline(Request{
		Model: clusterPlanTestModel(),
		Members: []membership.Member{
			clusterPlanTestMember("node-a", "node-a", "host-a", "http://host-a.local:52415"),
			clusterPlanTestMember("node-b", "node-b", "host-b", "http://host-b.local:52415"),
		},
		Now:        clusterPlanTestNow(),
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 2},
	})
	if !preview.Valid || preview.Mode != cluster.ExecutionModePipelineParallel || preview.StageCount != 2 {
		t.Fatalf("expected two-stage pipeline: %+v", preview)
	}
}
