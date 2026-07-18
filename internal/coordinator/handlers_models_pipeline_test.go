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
)

func TestRoutePreviewReportsOneStagePipeline(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.local:52415"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28

	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)
	request := httptest.NewRequest(http.MethodGet, api.RoutePreview+"?model=qwen2.5-coder-1.5b-q4&stage_count=1", nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}

	var preview clusterplan.RoutePreview
	if err := json.Unmarshal(response.Body.Bytes(), &preview); err != nil {
		t.Fatal(err)
	}
	if !preview.Valid || preview.Mode != cluster.ExecutionModePipelineParallel || preview.StageCount != 1 {
		t.Fatalf("unexpected preview: %+v", preview)
	}
}
