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

func TestChatCompletionsUsesSingleStagePipelineByDefault(t *testing.T) {
	stage := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.StageIndex != 0 || req.StageCount != 1 || req.LayerStart != 0 || req.LayerEnd != 28 {
			t.Fatalf("unexpected single-stage assignment: %+v", req.Metadata)
		}
		if req.PayloadKind != stagewire.PayloadKindText || !strings.Contains(string(req.Payload), "Say hello") {
			t.Fatalf("unexpected chat prompt: kind=%q payload=%q", req.PayloadKind, req.Payload)
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, 7)
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)
		metadata.Message = "hello"
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

	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":1
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get("X-JetsonFabric-Session-ID") == "" {
		t.Fatal("missing JetsonFabric session header")
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Choices) != 1 || decoded.Choices[0].Message.Content != "hello" {
		t.Fatalf("unexpected response: %+v", decoded)
	}
}

func TestChatCompletionsUsesDistributedPipeline(t *testing.T) {
	activation := make([]byte, 4*16*4)
	stage0 := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.PayloadKind != stagewire.PayloadKindText || !strings.Contains(string(req.Payload), "Explain CUDA") {
			t.Fatalf("unexpected chat prompt: kind=%q payload=%q", req.PayloadKind, req.Payload)
		}
		return stagewire.StageResponse{
			Metadata: responseMetadataForCoordinator(req, stagewire.PayloadKindActivation),
			Payload:  activation,
		}
	})
	defer stage0.Close()

	stage1 := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.PayloadKind != stagewire.PayloadKindActivation {
			t.Fatalf("expected activation, got %q", req.PayloadKind)
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, 42)
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)
		metadata.Message = "GPU answer"
		metadata.CompletionTokens = 1
		return stagewire.StageResponse{Metadata: metadata, Payload: payload}
	})
	defer stage1.Close()

	server := newLayerSplitTestServer(stage0.URL, stage1.URL)
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Explain CUDA"}],
		"max_tokens":1,
		"jetsonfabric":{"stage_count":2,"allow_colocated_stages":true}
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get("X-JetsonFabric-Session-ID") == "" {
		t.Fatal("missing JetsonFabric session header")
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Object != "chat.completion" || decoded.Model != "qwen2.5-coder-1.5b-q4" {
		t.Fatalf("unexpected response: %+v", decoded)
	}
	if len(decoded.Choices) != 1 || decoded.Choices[0].Message.Role != "assistant" || decoded.Choices[0].Message.Content != "GPU answer" {
		t.Fatalf("unexpected choices: %+v", decoded.Choices)
	}
	if decoded.Choices[0].FinishReason != "length" || decoded.Usage.CompletionTokens != 1 {
		t.Fatalf("unexpected completion metadata: %+v", decoded)
	}
}

func TestChatCompletionsReturnsOpenAIErrorShape(t *testing.T) {
	server := NewServer(coordinatorTestRegistry())
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{"messages":[]}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var decoded openAIErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Error.Type != "invalid_request_error" || decoded.Error.Code != "model_required" {
		t.Fatalf("unexpected error: %+v", decoded.Error)
	}
}
