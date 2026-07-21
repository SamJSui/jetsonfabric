package coordinator

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func TestChatCompletionsUsesSingleStagePipelineByDefault(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	generation := &recordingGenerationClient{events: []runtimebridge.GenerationEvent{
		{Type: "token", Token: uint32Pointer(7), Text: "hello", Index: 0},
		{Type: "done", FinishReason: "length", PromptTokens: 9, CompletionTokens: 1, SampledTokens: []uint32{7}, StageCalls: 1},
	}}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
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
	if response.Header().Get("X-JetsonFabric-Generation-Owner") != "runtime" {
		t.Fatalf("generation owner header=%q", response.Header().Get("X-JetsonFabric-Generation-Owner"))
	}
	if generation.calls != 1 || generation.nodeURL != member.APIURL {
		t.Fatalf("generation call did not target stage zero: calls=%d url=%q", generation.calls, generation.nodeURL)
	}
	if len(generation.request.Stages) != 1 || generation.request.Stages[0].StageIndex != 0 ||
		!strings.Contains(generation.request.Prompt, "Say hello") {
		t.Fatalf("unexpected runtime generation request: %+v", generation.request)
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
	generation := &recordingGenerationClient{events: []runtimebridge.GenerationEvent{
		{Type: "token", Token: uint32Pointer(42), Text: "GPU answer", Index: 0},
		{Type: "done", FinishReason: "length", PromptTokens: 11, CompletionTokens: 1, SampledTokens: []uint32{42}, StageCalls: 2, RemoteStageCalls: 1},
	}}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: membershipMembersForRun{
			{nodeID: "node-a", apiURL: "http://node-a.test"},
			{nodeID: "node-b", apiURL: "http://node-b.test"},
		}.members()}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)
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
	if generation.calls != 1 || generation.nodeURL != "http://node-a.test" {
		t.Fatalf("generation call did not target deterministic stage-zero leader: calls=%d url=%q", generation.calls, generation.nodeURL)
	}
	if len(generation.request.Stages) != 2 || generation.request.Stages[1].APIURL != "http://node-b.test" {
		t.Fatalf("runtime generation request omitted the peer plan: %+v", generation.request.Stages)
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

func TestChatCompletionsStreamsRuntimeTokens(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	generation := &recordingGenerationClient{events: []runtimebridge.GenerationEvent{
		{Type: "token", Token: uint32Pointer(7), Text: "hello", Index: 0},
		{Type: "token", Token: uint32Pointer(8), Text: " world", Index: 1},
		{Type: "done", FinishReason: "length", PromptTokens: 9, CompletionTokens: 2, SampledTokens: []uint32{7, 8}, StageCalls: 2},
	}}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":2,
		"stream":true
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status=%d content-type=%q body=%s", response.Code, response.Header().Get("Content-Type"), response.Body.String())
	}
	body := response.Body.String()
	for _, expected := range []string{`"role":"assistant"`, `"content":"hello"`, `"content":" world"`, `"finish_reason":"length"`, "data: [DONE]"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("stream omitted %q: %s", expected, body)
		}
	}
}

func TestChatCompletionsFlushesEachRuntimeTokenBeforeDone(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	reader, writer := io.Pipe()
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(pipeGenerationClient{reader: reader}),
	)
	apiServer := httptest.NewServer(server.Router())
	defer apiServer.Close()
	defer writer.Close()

	request, err := http.NewRequest(http.MethodPost, apiServer.URL+api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":1,
		"stream":true
	}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	firstEventWritten := make(chan error, 1)
	go func() {
		firstEventWritten <- json.NewEncoder(writer).Encode(runtimebridge.GenerationEvent{
			Type: "token", Token: uint32Pointer(7), Text: "hello", Index: 0,
		})
	}()
	response, err := apiServer.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if err := <-firstEventWritten; err != nil {
		t.Fatal(err)
	}
	stream := bufio.NewReader(response.Body)
	roleChunk, err := stream.ReadString('\n')
	if err != nil || !strings.Contains(roleChunk, `"role":"assistant"`) {
		t.Fatalf("role chunk was not flushed before runtime tokens: chunk=%q err=%v", roleChunk, err)
	}
	_, _ = stream.ReadString('\n')

	tokenChunk, err := stream.ReadString('\n')
	if err != nil || !strings.Contains(tokenChunk, `"content":"hello"`) {
		t.Fatalf("token chunk was not flushed before done: chunk=%q err=%v", tokenChunk, err)
	}
	_, _ = stream.ReadString('\n')

	if err := json.NewEncoder(writer).Encode(runtimebridge.GenerationEvent{
		Type: "done", FinishReason: "length", PromptTokens: 9, CompletionTokens: 1,
		SampledTokens: []uint32{7}, StageCalls: 1,
	}); err != nil {
		t.Fatal(err)
	}
	_ = writer.Close()
	remainder, err := io.ReadAll(stream)
	if err != nil || !strings.Contains(string(remainder), "data: [DONE]") {
		t.Fatalf("stream did not finish: body=%q err=%v", remainder, err)
	}
}

func TestStreamingChatCompletionReturnsHTTPErrorForImmediateRuntimeFailure(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	generation := &recordingGenerationClient{events: []runtimebridge.GenerationEvent{{
		Type: "error", Code: "model_not_loaded", Message: "runtime has no active model",
	}}}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":1,
		"stream":true
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.HasPrefix(response.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("status=%d content-type=%q body=%s", response.Code, response.Header().Get("Content-Type"), response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "model_not_loaded") || strings.Contains(response.Body.String(), `"role":"assistant"`) {
		t.Fatalf("unexpected immediate failure body: %s", response.Body.String())
	}
}

func TestChatCompletionsRejectsIncorrectRuntimeStageAccounting(t *testing.T) {
	member := membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0]
	member.Capabilities[cluster.CapabilityRuntimeStageIndex] = 0
	member.Capabilities[cluster.CapabilityRuntimeStageCount] = 1
	member.Capabilities[cluster.CapabilityRuntimeLayerStart] = 0
	member.Capabilities[cluster.CapabilityRuntimeLayerEnd] = 28
	generation := &recordingGenerationClient{events: []runtimebridge.GenerationEvent{
		{Type: "token", Token: uint32Pointer(7), Text: "hello", Index: 0},
		{Type: "done", FinishReason: "length", CompletionTokens: 1, SampledTokens: []uint32{7}, StageCalls: 0},
	}}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":1
	}`))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "stage call accounting") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestConsumeGenerationEventsRejectsCompletionCountMismatch(t *testing.T) {
	stream := strings.NewReader(
		"{\"type\":\"token\",\"token\":7,\"index\":0}\n" +
			"{\"type\":\"done\",\"finish_reason\":\"length\",\"completion_tokens\":0,\"sampled_tokens\":[7],\"stage_calls\":1}\n",
	)
	_, err := consumeGenerationEvents(stream, 1, nil)
	if err == nil || !strings.Contains(err.Error(), "completion token count") {
		t.Fatalf("unexpected result: %v", err)
	}
}

func TestConsumeGenerationEventsAccountsForNaturalStopPass(t *testing.T) {
	stream := strings.NewReader(
		"{\"type\":\"token\",\"token\":7,\"text\":\"hello\",\"index\":0}\n" +
			"{\"type\":\"done\",\"finish_reason\":\"stop\",\"completion_tokens\":1,\"sampled_tokens\":[7],\"stage_calls\":4,\"remote_stage_calls\":2}\n",
	)
	result, err := consumeGenerationEvents(stream, 2, nil)
	if err != nil {
		t.Fatalf("consume natural stop: %v", err)
	}
	if result.FinishReason != "stop" || result.GeneratedText != "hello" || result.StageCalls != 4 {
		t.Fatalf("unexpected natural-stop result: %+v", result)
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

type recordingGenerationClient struct {
	events  []runtimebridge.GenerationEvent
	err     error
	calls   int
	nodeURL string
	request runtimebridge.GenerationRequest
}

type pipeGenerationClient struct {
	reader io.ReadCloser
}

func (c pipeGenerationClient) Start(context.Context, string, runtimebridge.GenerationRequest) (runtimebridge.GenerationStream, error) {
	return runtimebridge.GenerationStream{
		Body:   c.reader,
		Header: http.Header{"Content-Type": []string{runtimebridge.GenerationContentType}},
	}, nil
}

func (c *recordingGenerationClient) Start(_ context.Context, nodeURL string, request runtimebridge.GenerationRequest) (runtimebridge.GenerationStream, error) {
	c.calls++
	c.nodeURL = nodeURL
	c.request = request
	if c.err != nil {
		return runtimebridge.GenerationStream{}, c.err
	}
	var body strings.Builder
	for _, event := range c.events {
		encoded, err := json.Marshal(event)
		if err != nil {
			return runtimebridge.GenerationStream{}, err
		}
		body.Write(encoded)
		body.WriteByte('\n')
	}
	return runtimebridge.GenerationStream{
		Body:   io.NopCloser(strings.NewReader(body.String())),
		Header: http.Header{"Content-Type": []string{runtimebridge.GenerationContentType}},
	}, nil
}

func uint32Pointer(value uint32) *uint32 {
	return &value
}
