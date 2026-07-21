package coordinator

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
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

func TestLayerSplitRunReportsActivationHandoff(t *testing.T) {
	activation := make([]byte, 4*16*4)
	for i := range activation {
		activation[i] = byte((i * 31) % 251)
	}
	stage0 := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindActivation)
		return stagewire.StageResponse{Metadata: metadata, Payload: activation}
	})
	defer stage0.Close()
	stage1 := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.PayloadKind != stagewire.PayloadKindActivation || crc32.ChecksumIEEE(req.Payload) != crc32.ChecksumIEEE(activation) {
			t.Fatalf("activation changed: %+v", req.Metadata)
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, crc32.ChecksumIEEE(req.Payload))
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)
		metadata.CompletionTokens = 1
		return stagewire.StageResponse{Metadata: metadata, Payload: payload}
	})
	defer stage1.Close()

	server := newLayerSplitTestServer(stage0.URL, stage1.URL)
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(`{
		"request_id":"run-1",
		"model":"qwen2.5-coder-1.5b-q4",
		"payload":"prompt",
		"max_tokens":1,
		"stage_count":2,
		"allow_colocated_stages":true
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
	if decoded.InterStagePayloadKind != stagewire.PayloadKindActivation || decoded.Result.PayloadKind != stagewire.PayloadKindSampledToken || decoded.Result.SampledToken == nil {
		t.Fatalf("unexpected response: %+v", decoded)
	}
	if decoded.RuntimeIdentity.ModelSHA256 != coordinatorTestModelSHA256 {
		t.Fatalf("runtime identity was not returned: %+v", decoded.RuntimeIdentity)
	}
	if decoded.Result.RequestID != "run-1" || decoded.Result.SessionID == "" {
		t.Fatalf("unexpected result IDs: %+v", decoded.Result)
	}
	if decoded.Result.Stages[0].PayloadCRC32Out != decoded.Result.Stages[1].PayloadCRC32In {
		t.Fatalf("checksum changed across nodes: %+v", decoded.Result.Stages)
	}
}

func TestLayerSplitRunRejectsColocatedPlanByDefault(t *testing.T) {
	stage0 := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected stage call") }))
	defer stage0.Close()
	stage1 := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected stage call") }))
	defer stage1.Close()
	server := newLayerSplitTestServer(stage0.URL, stage1.URL)
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4","payload":"prompt","stage_count":2
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestLayerSplitRunReturnsStageFailure(t *testing.T) {
	stage0 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := stagewire.Decode(r.Body)
		if err != nil {
			t.Fatalf("decode frame: %v", err)
		}
		if req.Operation != stagewire.OperationCloseSession {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime_down", "message": "unavailable"})
			return
		}
		encoded, err := stagewire.Marshal(stagewire.StageResponse{
			Metadata: responseMetadataForCoordinator(req, stagewire.PayloadKindText),
		})
		if err != nil {
			t.Fatalf("encode close response: %v", err)
		}
		w.Header().Set("Content-Type", stagewire.ContentType)
		_, _ = w.Write(encoded)
	}))
	defer stage0.Close()

	stage1Executed := false
	stage1 := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		stage1Executed = true
		return stagewire.StageResponse{Metadata: responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)}
	})
	defer stage1.Close()

	server := newLayerSplitTestServer(stage0.URL, stage1.URL)
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4","payload":"prompt","allow_colocated_stages":true
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || stage1Executed {
		t.Fatalf("status=%d stage1_executed=%v body=%s", response.Code, stage1Executed, response.Body.String())
	}
}

func TestLayerSplitRunRejectsMismatchedArtifacts(t *testing.T) {
	stage0 := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected stage call") }))
	defer stage0.Close()
	stage1 := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected stage call") }))
	defer stage1.Close()
	members := membershipMembersForRun{
		{nodeID: "node-a", apiURL: stage0.URL, modelSHA256: strings.Repeat("a", 64)},
		{nodeID: "node-b", apiURL: stage1.URL, modelSHA256: strings.Repeat("b", 64)},
	}.members()
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: members}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4","payload":"prompt","stage_count":2,"allow_colocated_stages":true
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), "matching engine, model artifact") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestLayerSplitRunRejectsNonPositiveStageCount(t *testing.T) {
	for _, stageCount := range []int{0, -1} {
		t.Run(fmt.Sprintf("stage_count_%d", stageCount), func(t *testing.T) {
			server := NewServer(coordinatorTestRegistry())
			request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(fmt.Sprintf(`{
				"model":"qwen2.5-coder-1.5b-q4","payload":"prompt","stage_count":%d
			}`, stageCount)))
			response := httptest.NewRecorder()
			server.Router().ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func newCoordinatorFrameServer(t *testing.T, run func(stagewire.StageRequest) stagewire.StageResponse) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := stagewire.Decode(r.Body)
		if err != nil {
			t.Fatalf("decode frame: %v", err)
		}
		var response stagewire.StageResponse
		if req.Operation == stagewire.OperationCloseSession {
			response = stagewire.StageResponse{Metadata: responseMetadataForCoordinator(req, stagewire.PayloadKindText)}
		} else {
			response = run(req)
		}
		encoded, err := stagewire.Marshal(response)
		if err != nil {
			t.Fatalf("encode response: %v", err)
		}
		w.Header().Set("Content-Type", stagewire.ContentType)
		_, _ = w.Write(encoded)
	}))
}

func responseMetadataForCoordinator(req stagewire.StageRequest, kind stagewire.PayloadKind) stagewire.Metadata {
	metadata := req.Metadata
	metadata.PayloadKind = kind
	metadata.Encoding = ""
	metadata.DType = ""
	metadata.Shape = nil
	metadata.ByteOrder = ""
	metadata.Layout = ""
	if kind == stagewire.PayloadKindText {
		metadata.Encoding = "utf-8"
	}
	if kind == stagewire.PayloadKindActivation {
		metadata.DType = "f32"
		metadata.Shape = []int64{4, 16}
		metadata.ByteOrder = "little"
		metadata.Layout = "row_major"
	}
	if kind == stagewire.PayloadKindSampledToken {
		metadata.DType = "u32"
		metadata.Shape = []int64{1}
		metadata.ByteOrder = "little"
		metadata.Layout = "row_major"
	}
	return metadata
}

func newLayerSplitTestServer(stage0URL, stage1URL string) *Server {
	return NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: membershipMembersForRun{
			{nodeID: "node-a", apiURL: stage0URL},
			{nodeID: "node-b", apiURL: stage1URL},
		}.members()}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
	)
}

const coordinatorTestModelSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

type membershipMemberForRun struct {
	nodeID      string
	apiURL      string
	modelSHA256 string
}

type membershipMembersForRun []membershipMemberForRun

func (items membershipMembersForRun) members() []membership.Member {
	members := make([]membership.Member, 0, len(items))
	for _, item := range items {
		member := coordinatorTestMember(item.nodeID, "dopey", item.apiURL, "http://127.0.0.1:9090")
		modelSHA256 := item.modelSHA256
		if modelSHA256 == "" {
			modelSHA256 = coordinatorTestModelSHA256
		}
		member.Capabilities[cluster.CapabilityRuntimeEngine] = string(cluster.EngineLlamaCPP)
		member.Capabilities[cluster.CapabilityRuntimeModelID] = "qwen2.5-coder-1.5b-q4"
		member.Capabilities[cluster.CapabilityRuntimeModelSHA256] = modelSHA256
		member.Capabilities[cluster.CapabilityRuntimeComputeBackend] = string(cluster.ComputeBackendCPU)
		member.Capabilities[cluster.CapabilityRuntimeExecutionMode] = string(cluster.ExecutionModePipelineParallel)
		members = append(members, member)
	}
	return members
}
