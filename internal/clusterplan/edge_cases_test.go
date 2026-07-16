package clusterplan

import (
	"fmt"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestAssignLayerRangesBalancesEveryValidStageCount(t *testing.T) {
	const layerCount = 28
	for stageCount := 1; stageCount <= layerCount; stageCount++ {
		t.Run(fmt.Sprintf("stages_%d", stageCount), func(t *testing.T) {
			ranges := AssignLayerRanges(layerCount, stageCount)
			if len(ranges) != stageCount {
				t.Fatalf("len(ranges) = %d, want %d", len(ranges), stageCount)
			}
			minWidth, maxWidth := layerCount, 0
			expectedStart := 0
			for index, item := range ranges {
				if item.Start != expectedStart || item.End <= item.Start {
					t.Fatalf("range %d is not positive and contiguous: %+v", index, item)
				}
				width := item.End - item.Start
				if width < minWidth {
					minWidth = width
				}
				if width > maxWidth {
					maxWidth = width
				}
				expectedStart = item.End
			}
			if expectedStart != layerCount {
				t.Fatalf("ranges end at %d, want %d", expectedStart, layerCount)
			}
			if maxWidth-minWidth > 1 {
				t.Fatalf("unbalanced ranges: min=%d max=%d ranges=%+v", minWidth, maxWidth, ranges)
			}
		})
	}
}

func TestAssignLayerRangesRejectsInvalidDimensions(t *testing.T) {
	tests := []struct {
		layers int
		stages int
	}{
		{layers: 28, stages: 0},
		{layers: 28, stages: -1},
		{layers: 0, stages: 1},
		{layers: -1, stages: 1},
		{layers: 3, stages: 4},
	}
	for _, test := range tests {
		if got := AssignLayerRanges(test.layers, test.stages); got != nil {
			t.Fatalf("AssignLayerRanges(%d, %d) = %+v, want nil", test.layers, test.stages, got)
		}
	}
}

func TestPlanRejectsInvalidRequestedStageCounts(t *testing.T) {
	tests := []struct {
		name       string
		layerCount int
		stageCount int
		want       Reason
	}{
		{name: "negative", layerCount: 28, stageCount: -1, want: ReasonInvalidStageCount},
		{name: "more stages than layers", layerCount: 3, stageCount: 4, want: ReasonStageCountExceedsLayers},
		{name: "missing layer metadata", layerCount: 0, stageCount: 2, want: ReasonInvalidLayerCount},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := clusterPlanTestModel()
			model.LayerCount = test.layerCount
			preview := Preview(Request{Model: model, Members: edgeCaseMembers(4), Now: clusterPlanTestNow(), StaleAfter: time.Minute, Policy: Policy{StageCount: test.stageCount}})
			if preview.Valid || preview.Reason != test.want {
				t.Fatalf("preview = %+v, want reason %q", preview, test.want)
			}
		})
	}
}

func TestPlanRejectsIncompleteAdvertisedAssignments(t *testing.T) {
	tests := []struct {
		name        string
		assignments [][4]int
	}{
		{name: "duplicate index", assignments: [][4]int{{0, 2, 0, 14}, {0, 2, 14, 28}}},
		{name: "gap", assignments: [][4]int{{0, 2, 0, 13}, {1, 2, 14, 28}}},
		{name: "overlap", assignments: [][4]int{{0, 2, 0, 15}, {1, 2, 14, 28}}},
		{name: "ends early", assignments: [][4]int{{0, 2, 0, 14}, {1, 2, 14, 27}}},
		{name: "extends past model", assignments: [][4]int{{0, 2, 0, 14}, {1, 2, 14, 29}}},
		{name: "mixed counts", assignments: [][4]int{{0, 2, 0, 14}, {1, 3, 14, 28}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			members := make([]membership.Member, 0, len(test.assignments))
			for index, assignment := range test.assignments {
				member := clusterPlanTestMember(fmt.Sprintf("node-%d", index), fmt.Sprintf("node-%d", index), fmt.Sprintf("host-%d", index), fmt.Sprintf("http://host-%d.local:52415", index))
				advertiseRuntimeAssignment(&member, assignment[0], assignment[1], assignment[2], assignment[3])
				members = append(members, member)
			}
			preview := Preview(Request{Model: clusterPlanTestModel(), Members: members, Now: clusterPlanTestNow(), StaleAfter: time.Minute})
			if preview.Valid || preview.Reason != ReasonIncompleteRuntimeAssignments {
				t.Fatalf("preview = %+v, want incomplete assignments", preview)
			}
		})
	}
}

func TestPlanFallsBackOnlyToVerifiedFullModelReplica(t *testing.T) {
	partial := clusterPlanTestMember("partial", "partial", "host-a", "http://host-a.local:52415")
	advertiseRuntimeAssignment(&partial, 0, 2, 0, 14)
	full := clusterPlanTestMember("full", "full", "host-b", "http://host-b.local:52415")
	advertiseRuntimeAssignment(&full, 0, 1, 0, 28)

	preview := Preview(Request{Model: clusterPlanTestModel(), Members: []membership.Member{partial, full}, Now: clusterPlanTestNow(), StaleAfter: time.Minute})
	if !preview.Valid || preview.Mode != cluster.ExecutionModeDataParallel || len(preview.Stages) != 1 {
		t.Fatalf("expected verified full-model fallback: %+v", preview)
	}
	if preview.Stages[0].NodeID != "full" || preview.Stages[0].StageCount != 1 || preview.Stages[0].LayerStart != 0 || preview.Stages[0].LayerEnd != 28 {
		t.Fatalf("unexpected fallback stage: %+v", preview.Stages[0])
	}
}

func TestValidateStagePlanRejectsMismatchedMetadata(t *testing.T) {
	tests := []struct {
		name   string
		stages []Stage
		want   Reason
	}{
		{name: "wrong count", stages: []Stage{{StageIndex: 0, StageCount: 3, LayerStart: 0, LayerEnd: 14}, {StageIndex: 1, StageCount: 3, LayerStart: 14, LayerEnd: 28}}, want: ReasonInvalidStageIndices},
		{name: "wrong index", stages: []Stage{{StageIndex: 0, StageCount: 2, LayerStart: 0, LayerEnd: 14}, {StageIndex: 2, StageCount: 2, LayerStart: 14, LayerEnd: 28}}, want: ReasonInvalidStageIndices},
		{name: "gap", stages: []Stage{{StageIndex: 0, StageCount: 2, LayerStart: 0, LayerEnd: 13}, {StageIndex: 1, StageCount: 2, LayerStart: 14, LayerEnd: 28}}, want: ReasonInvalidLayerRanges},
		{name: "zero width", stages: []Stage{{StageIndex: 0, StageCount: 2, LayerStart: 0, LayerEnd: 28}, {StageIndex: 1, StageCount: 2, LayerStart: 28, LayerEnd: 28}}, want: ReasonInvalidLayerRanges},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := validateStagePlan(test.stages, 28); got != test.want {
				t.Fatalf("validateStagePlan() = %q, want %q", got, test.want)
			}
		})
	}
}

func edgeCaseMembers(count int) []membership.Member {
	members := make([]membership.Member, 0, count)
	for index := 0; index < count; index++ {
		members = append(members, clusterPlanTestMember(fmt.Sprintf("node-%d", index), fmt.Sprintf("node-%d", index), fmt.Sprintf("host-%d", index), fmt.Sprintf("http://host-%d.local:52415", index)))
	}
	return members
}
