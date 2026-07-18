package coordinator

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestLayerSplitRunSupportsOneStagePipeline(t *testing.T) {
	stage := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.StageIndex != 0 || req.StageCount != 1 || req.LayerStart != 0 || req.LayerEnd != 28 {
			t.Fatalf("unexpected assignment: %+v", req.Metadata)
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, 9)
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)
		metadata.Message = "token"
		metadata.CompletionTokens = 1
		return stagewire.StageResponse{Metadata: metadata, Payload: payload}
	})
	defer stage.Close()

	member := membershipMembersForRun{{nodeID: "node-a", apiURL: stage.URL}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)

	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(`{
		"request_id":"single-run",
		"model":"qwen2.5-coder-1.5b-q4",
		"payload":"hello",
		"max_tokens":1,
		"stage_count":1
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}

	var decoded layerSplitRunResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Plan.Mode != cluster.ExecutionModePipelineParallel || decoded.Plan.StageCount != 1 {
		t.Fatalf("unexpected plan: %+v", decoded.Plan)
	}
	if decoded.Result.PayloadKind != stagewire.PayloadKindSampledToken || decoded.Result.SampledToken == nil {
		t.Fatalf("unexpected result: %+v", decoded.Result)
	}
}
