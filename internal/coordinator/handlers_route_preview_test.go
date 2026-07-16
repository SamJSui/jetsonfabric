package coordinator

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
)

func TestRoutePreviewUsesMembershipSource(t *testing.T) {
	members := []membership.Member{
		coordinatorTestMember("node-a", "dopey", "http://dopey.local:52415", "http://127.0.0.1:9001"),
		coordinatorTestMember("node-b", "grumpy", "http://grumpy.local:52415", "http://127.0.0.1:9002"),
	}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: members}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)

	request := httptest.NewRequest(http.MethodGet, api.PathRoutePreview+"?model=qwen2.5-coder-1.5b-q4", nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	preview := decodeClusterPlanPreview(t, response)
	if !preview.Valid || preview.Topology != clusterplan.TopologyDistributed || preview.StageCount != 2 || preview.PhysicalHostCount != 2 {
		t.Fatalf("unexpected preview: %+v", preview)
	}
	for _, stage := range preview.Stages {
		if stage.APIURL == "http://127.0.0.1:9001" || stage.APIURL == "http://127.0.0.1:9002" {
			t.Fatalf("stage used runtime URL: %+v", stage)
		}
	}
}

func TestRoutePreviewAllowsExplicitColocatedStages(t *testing.T) {
	members := []membership.Member{
		coordinatorTestMember("node-a", "dopey", "http://dopey-a.local:52415", "http://127.0.0.1:9001"),
		coordinatorTestMember("node-b", "dopey", "http://dopey-b.local:52416", "http://127.0.0.1:9002"),
	}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: members}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)

	request := httptest.NewRequest(http.MethodGet, api.PathRoutePreview+"?model=qwen2.5-coder-1.5b-q4&allow_colocated_stages=true", nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	preview := decodeClusterPlanPreview(t, response)
	if !preview.Valid || preview.Topology != clusterplan.TopologyColocated || preview.PhysicalHostCount != 1 || preview.StageCount != 2 || len(preview.Warnings) == 0 {
		t.Fatalf("unexpected colocated preview: %+v", preview)
	}
}

func TestRoutePreviewAcceptsRequestedStageCount(t *testing.T) {
	members := []membership.Member{
		coordinatorTestMember("node-a", "a", "http://a.local:52415", "http://127.0.0.1:9001"),
		coordinatorTestMember("node-b", "b", "http://b.local:52415", "http://127.0.0.1:9002"),
		coordinatorTestMember("node-c", "c", "http://c.local:52415", "http://127.0.0.1:9003"),
	}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: members}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)
	request := httptest.NewRequest(http.MethodGet, api.PathRoutePreview+"?model=qwen2.5-coder-1.5b-q4&stage_count=3", nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	preview := decodeClusterPlanPreview(t, response)
	if !preview.Valid || preview.StageCount != 3 || len(preview.Stages) != 3 {
		t.Fatalf("expected three-stage preview: %+v", preview)
	}
}

func TestRoutePreviewWithoutMembershipCandidatesIsInvalid(t *testing.T) {
	server := NewServer(coordinatorTestRegistry(), WithClock(func() time.Time { return coordinatorTestNow() }))
	request := httptest.NewRequest(http.MethodGet, api.PathRoutePreview+"?model=qwen2.5-coder-1.5b-q4", nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	preview := decodeClusterPlanPreview(t, response)
	if preview.Valid || preview.Reason != clusterplan.ReasonNoEligibleMembers {
		t.Fatalf("unexpected preview without membership candidates: %+v", preview)
	}
}

func decodeClusterPlanPreview(t *testing.T, response *httptest.ResponseRecorder) clusterplan.RoutePreview {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("expected OK, got %d: %s", response.Code, response.Body.String())
	}
	var preview clusterplan.RoutePreview
	if err := json.Unmarshal(response.Body.Bytes(), &preview); err != nil {
		t.Fatalf("decode preview: %v", err)
	}
	return preview
}

type staticMemberSource struct {
	members []membership.Member
}

func (s staticMemberSource) List() []membership.Member {
	return append([]membership.Member(nil), s.members...)
}

func coordinatorTestRegistry() modelregistry.Registry {
	return modelregistry.Registry{Models: []cluster.ModelProfile{{
		ID: "qwen2.5-coder-1.5b-q4", LayerCount: 28, MinMemoryGB: 3,
		SupportedEngines: []cluster.Engine{cluster.EngineLlamaCPP},
		PlacementModes:   []cluster.ExecutionMode{cluster.ExecutionModeDataParallel, cluster.ExecutionModePipelineParallel},
	}}}
}

func coordinatorTestMember(id, host, apiURL, runtimeURL string) membership.Member {
	now := coordinatorTestNow()
	return membership.Member{
		ClusterID: "home-lab", NodeID: id, NodeName: id, Hostname: host,
		Role: membership.NodeRoleJetson, APIURL: apiURL, RuntimeURL: runtimeURL,
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:        8.0,
			cluster.CapabilityComputeBackends: []string{string(cluster.ComputeBackendCPU), string(cluster.ComputeBackendCUDA)},
		},
		StartedAt: now.Add(-time.Minute), LastSeen: now,
	}
}

func coordinatorTestNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}

func TestRoutePreviewRejectsInvalidStageCountQuery(t *testing.T) {
	for _, value := range []string{"0", "-1", "not-a-number"} {
		t.Run(value, func(t *testing.T) {
			server := NewServer(coordinatorTestRegistry(), WithMembershipSource(staticMemberSource{members: []membership.Member{coordinatorTestMember("node-a", "dopey", "http://dopey.local:52415", "http://127.0.0.1:9001")}}, time.Minute), WithClock(func() time.Time { return coordinatorTestNow() }))
			request := httptest.NewRequest(http.MethodGet, api.PathRoutePreview+"?model=qwen2.5-coder-1.5b-q4&stage_count="+value, nil)
			response := httptest.NewRecorder()
			server.Router().ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("expected OK preview response, got %d: %s", response.Code, response.Body.String())
			}
			preview := decodeClusterPlanPreview(t, response)
			if preview.Valid || preview.Reason != clusterplan.ReasonInvalidStageCount {
				t.Fatalf("unexpected preview: %+v", preview)
			}
		})
	}
}
