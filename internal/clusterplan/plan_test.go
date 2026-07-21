package clusterplan

import (
	"fmt"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestPlanUsesAPIURLNotRuntimeURL(t *testing.T) {
	now := clusterPlanTestNow()
	member := clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey.local:52415")
	member.RuntimeURL = "http://127.0.0.1:9090"

	preview := Preview(Request{Model: clusterPlanTestModel(), Members: []membership.Member{member}, Now: now, StaleAfter: time.Minute})
	if !preview.Valid || len(preview.Stages) != 1 {
		t.Fatalf("expected valid one-stage preview: %+v", preview)
	}
	if preview.Stages[0].APIURL != member.APIURL || preview.Stages[0].APIURL == member.RuntimeURL {
		t.Fatalf("planner must use node API URL: %+v", preview.Stages[0])
	}
}

func TestPlanSkipsStaleMembers(t *testing.T) {
	now := clusterPlanTestNow()
	fresh := clusterPlanTestMember("fresh", "fresh", "dopey", "http://fresh.local:52415")
	stale := clusterPlanTestMember("stale", "stale", "grumpy", "http://stale.local:52415")
	stale.LastSeen = now.Add(-2 * time.Minute)

	preview := Preview(Request{Model: clusterPlanTestModel(), Members: []membership.Member{fresh, stale}, Now: now, StaleAfter: time.Minute})
	if !preview.Valid || len(preview.Placements) != 1 || preview.Placements[0].NodeID != "fresh" {
		t.Fatalf("expected only fresh placement: %+v", preview)
	}
}

func TestPlanUsesCountsForOneEligibleMember(t *testing.T) {
	preview := Preview(Request{
		Model:      clusterPlanTestModel(),
		Members:    []membership.Member{clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey.local:52415")},
		Now:        clusterPlanTestNow(),
		StaleAfter: time.Minute,
	})
	if !preview.Valid || preview.Mode != cluster.ExecutionModeDataParallel {
		t.Fatalf("expected data_parallel preview: %+v", preview)
	}
	if preview.Topology != TopologyColocated || preview.StageCount != 1 || preview.LogicalNodeCount != 1 || preview.PhysicalHostCount != 1 {
		t.Fatalf("unexpected topology counts: %+v", preview)
	}
	if len(preview.Stages) != 1 || preview.Stages[0].StageIndex != 0 || preview.Stages[0].StageCount != 1 {
		t.Fatalf("unexpected stage arithmetic: %+v", preview.Stages)
	}
}

func TestPlanReturnsDistributedTopologyForDistinctHosts(t *testing.T) {
	preview := Preview(Request{
		Model: clusterPlanTestModel(),
		Members: []membership.Member{
			clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey.local:52415"),
			clusterPlanTestMember("node-b", "grumpy-a", "grumpy", "http://grumpy.local:52415"),
		},
		Now: clusterPlanTestNow(), StaleAfter: time.Minute,
	})
	if !preview.Valid || preview.Mode != cluster.ExecutionModePipelineParallel || preview.Topology != TopologyDistributed {
		t.Fatalf("expected distributed pipeline preview: %+v", preview)
	}
	if preview.StageCount != 2 || preview.LogicalNodeCount != 2 || preview.PhysicalHostCount != 2 {
		t.Fatalf("unexpected counts: %+v", preview)
	}
	assertRange(t, preview.Stages[0], 0, 14)
	assertRange(t, preview.Stages[1], 14, 28)
}

func TestPlanSupportsRequestedStageCount(t *testing.T) {
	members := make([]membership.Member, 0, 3)
	for i := 0; i < 3; i++ {
		members = append(members, clusterPlanTestMember(
			fmt.Sprintf("node-%d", i),
			fmt.Sprintf("node-%d", i),
			fmt.Sprintf("host-%d", i),
			fmt.Sprintf("http://host-%d.local:52415", i),
		))
	}
	preview := Preview(Request{
		Model: clusterPlanTestModel(), Members: members, Now: clusterPlanTestNow(), StaleAfter: time.Minute,
		Policy: Policy{StageCount: 3},
	})
	if !preview.Valid || preview.StageCount != 3 || preview.PhysicalHostCount != 3 {
		t.Fatalf("expected three-stage preview: %+v", preview)
	}
	want := []LayerRange{{0, 10}, {10, 19}, {19, 28}}
	for i, stage := range preview.Stages {
		if stage.StageIndex != i || stage.StageCount != 3 || stage.LayerStart != want[i].Start || stage.LayerEnd != want[i].End {
			t.Fatalf("unexpected stage %d: %+v", i, stage)
		}
	}
}

func TestPlanUsesEveryEligibleNodeWhenStageCountIsAutomatic(t *testing.T) {
	members := make([]membership.Member, 0, 3)
	for i := 0; i < 3; i++ {
		members = append(members, clusterPlanTestMember(
			fmt.Sprintf("node-%d", i),
			fmt.Sprintf("node-%d", i),
			fmt.Sprintf("host-%d", i),
			fmt.Sprintf("http://host-%d.local:52415", i),
		))
	}
	preview := Preview(Request{
		Model: clusterPlanTestModel(), Members: members,
		Now: clusterPlanTestNow(), StaleAfter: time.Minute,
	})
	if !preview.Valid || preview.StageCount != 3 || preview.PhysicalHostCount != 3 {
		t.Fatalf("automatic placement did not use every eligible node: %+v", preview)
	}
}

func TestPlanRejectsUnavailableRequestedStageCount(t *testing.T) {
	preview := Preview(Request{
		Model: clusterPlanTestModel(),
		Members: []membership.Member{
			clusterPlanTestMember("node-a", "node-a", "host-a", "http://host-a.local:52415"),
			clusterPlanTestMember("node-b", "node-b", "host-b", "http://host-b.local:52415"),
		},
		Now: clusterPlanTestNow(), StaleAfter: time.Minute, Policy: Policy{StageCount: 3},
	})
	if preview.Valid || preview.Reason != ReasonInsufficientStages {
		t.Fatalf("expected insufficient stage error: %+v", preview)
	}
}

func TestPlanRejectsColocatedStagesWhenPolicyDisallows(t *testing.T) {
	preview := Preview(Request{
		Model: clusterPlanTestModel(),
		Members: []membership.Member{
			clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey-a.local:52415"),
			clusterPlanTestMember("node-b", "dopey-b", "dopey", "http://dopey-b.local:52416"),
		},
		Now: clusterPlanTestNow(), StaleAfter: time.Minute,
	})
	if preview.Valid || preview.Reason != ReasonColocatedStagesDisallowed || preview.Topology != TopologyColocated {
		t.Fatalf("expected rejected colocated preview: %+v", preview)
	}
}

func TestPlanAllowsColocatedStagesWithExplicitPolicy(t *testing.T) {
	preview := Preview(Request{
		Model: clusterPlanTestModel(),
		Members: []membership.Member{
			clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey-a.local:52415"),
			clusterPlanTestMember("node-b", "dopey-b", "dopey", "http://dopey-b.local:52416"),
		},
		Now: clusterPlanTestNow(), StaleAfter: time.Minute, Policy: Policy{AllowColocatedStages: true},
	})
	if !preview.Valid || preview.Topology != TopologyColocated || preview.PhysicalHostCount != 1 || len(preview.Warnings) == 0 {
		t.Fatalf("expected allowed colocated preview: %+v", preview)
	}
}

func TestPlanUsesAdvertisedRuntimeStageAssignments(t *testing.T) {
	stage0 := clusterPlanTestMember("node-a", "dopey-stage0", "dopey", "http://dopey-a.local:52415")
	advertiseRuntimeAssignment(&stage0, 0, 2, 0, 14)
	stage1 := clusterPlanTestMember("node-b", "dopey-stage1", "dopey", "http://dopey-b.local:52416")
	advertiseRuntimeAssignment(&stage1, 1, 2, 14, 28)

	preview := Preview(Request{
		Model: clusterPlanTestModel(), Members: []membership.Member{stage1, stage0}, Now: clusterPlanTestNow(), StaleAfter: time.Minute,
		Policy: Policy{AllowColocatedStages: true},
	})
	if !preview.Valid || len(preview.Stages) != 2 {
		t.Fatalf("expected advertised pipeline: %+v", preview)
	}
	if preview.Stages[0].NodeID != "node-a" || preview.Stages[1].NodeID != "node-b" {
		t.Fatalf("unexpected order: %+v", preview.Stages)
	}
}

func TestPlanDoesNotInventRangesForFullModelAssignments(t *testing.T) {
	first := clusterPlanTestMember("node-a", "dopey-a", "dopey", "http://dopey-a.local:52415")
	advertiseRuntimeAssignment(&first, 0, 1, 0, 28)
	second := clusterPlanTestMember("node-b", "dopey-b", "dopey", "http://dopey-b.local:52416")
	advertiseRuntimeAssignment(&second, 0, 1, 0, 28)

	preview := Preview(Request{
		Model: clusterPlanTestModel(), Members: []membership.Member{first, second}, Now: clusterPlanTestNow(), StaleAfter: time.Minute,
		Policy: Policy{AllowColocatedStages: true},
	})
	if !preview.Valid || preview.Mode != cluster.ExecutionModeDataParallel || preview.StageCount != 1 {
		t.Fatalf("expected one full-model replica: %+v", preview)
	}
	assertRange(t, preview.Stages[0], 0, 28)
}

func TestAssignLayerRangesCoversAllLayersWithoutOverlap(t *testing.T) {
	ranges := AssignLayerRanges(28, 3)
	want := []LayerRange{{0, 10}, {10, 19}, {19, 28}}
	if len(ranges) != len(want) {
		t.Fatalf("unexpected ranges: %+v", ranges)
	}
	for i := range want {
		if ranges[i] != want[i] {
			t.Fatalf("range %d = %+v, want %+v", i, ranges[i], want[i])
		}
	}
}

func assertRange(t *testing.T, stage Stage, start, end int) {
	t.Helper()
	if stage.LayerStart != start || stage.LayerEnd != end {
		t.Fatalf("range = [%d:%d], want [%d:%d]", stage.LayerStart, stage.LayerEnd, start, end)
	}
}

func clusterPlanTestModel() cluster.ModelProfile {
	return cluster.ModelProfile{
		ID: "qwen2.5-coder-1.5b-q4", LayerCount: 28, MinMemoryGB: 3,
		PlacementModes:   []cluster.ExecutionMode{cluster.ExecutionModeDataParallel, cluster.ExecutionModePipelineParallel},
		SupportedEngines: []cluster.Engine{cluster.EngineLlamaCPP},
	}
}

func clusterPlanTestMember(id, name, hostname, apiURL string) membership.Member {
	now := clusterPlanTestNow()
	return membership.Member{
		ClusterID: "home-lab", NodeID: id, NodeName: name, Hostname: hostname,
		Role: membership.NodeRoleJetson, APIURL: apiURL,
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:        8.0,
			cluster.CapabilityDeviceClass:     string(cluster.DeviceClassJetson),
			cluster.CapabilityComputeBackends: []string{string(cluster.ComputeBackendCPU), string(cluster.ComputeBackendCUDA)},
		},
		StartedAt: now.Add(-time.Minute), LastSeen: now,
	}
}

func advertiseRuntimeAssignment(member *membership.Member, stageIndex, stageCount, layerStart, layerEnd int) {
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = stageIndex
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = stageCount
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = layerStart
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = layerEnd
}

func clusterPlanTestNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}
