package control

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/agent"
	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

func TestChatCompletionsRoutesToSingleNodeBackend(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathChatCompletions {
			t.Fatalf("unexpected backend path: %s", r.URL.Path)
		}
		var req chat.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode backend request: %v", err)
		}
		if req.Model != "qwen2.5-coder-1.5b-q4" {
			t.Fatalf("unexpected model routed to backend: %s", req.Model)
		}
		writeJSON(w, http.StatusOK, chat.CompletionResponse{
			ID:      "chatcmpl-test",
			Object:  "chat.completion",
			Created: 123,
			Model:   req.Model,
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "hello from jetson",
					},
					FinishReason: "stop",
				},
			},
			Usage: &chat.Usage{CompletionTokens: 3, TotalTokens: 8},
		})
	}))
	defer backend.Close()

	recorder := &recordingRecorder{}
	handler := NewServer(
		"dev-token",
		testRegistry(),
		WithBenchmarkRecorder(recorder),
		WithClock(func() time.Time { return time.Unix(100, 0).UTC() }),
	).Router()
	registerNode(t, handler, cluster.HeartbeatRequest{
		NodeName: "jetson-01",
		Arch:     "arm64",
		OS:       "linux",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:        8.0,
			cluster.CapabilityComputeBackends: []any{cluster.ComputeBackendCPU, cluster.ComputeBackendCUDA},
		},
		Metrics: map[string]any{
			cluster.MetricTemperatureC: 42.5,
		},
		Engines: []cluster.EngineEndpoint{
			{
				InstanceID:       cluster.Llm,
				Engine:           cluster.EngineLlamaCPP,
				BaseURL:          backend.URL,
				Models:           []string{"qwen2.5-coder-1.5b-q4"},
				OpenAICompatible: true,
			},
		},
	})

	response := postChat(t, handler, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if response.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", response.Code, response.Body.String())
	}
	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode chat response: %v", err)
	}
	choice := firstChoice(t, decoded)
	if choice.Message.Content != "hello from jetson" {
		t.Fatalf("unexpected response content: %s", choice.Message.Content)
	}
	if decoded.Route == nil {
		t.Fatal("expected route metadata")
	}
	if decoded.Route.Mode != cluster.RouteModeSingleNode || decoded.Route.NodeName != "jetson-01" || decoded.Route.BackendID != cluster.BackendIDLlamaLocal {
		t.Fatalf("unexpected route metadata: %+v", decoded.Route)
	}
	if len(recorder.records) != 1 {
		t.Fatalf("expected one benchmark record, got %d", len(recorder.records))
	}
	record := firstRecord(t, recorder.records)
	if record.ModelID != "qwen2.5-coder-1.5b-q4" || record.NodeName != "jetson-01" || record.RouteMode != cluster.RouteModeSingleNode {
		t.Fatalf("unexpected benchmark record: %+v", record)
	}
	if record.MemoryGB == nil || *record.MemoryGB != 8.0 {
		t.Fatalf("expected memory benchmark field, got %+v", record.MemoryGB)
	}
	if record.TemperatureC == nil || *record.TemperatureC != 42.5 {
		t.Fatalf("expected temperature benchmark field, got %+v", record.TemperatureC)
	}
}

func TestServerRunsOverHTTPForSingleNodeChatFlow(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathChatCompletions {
			t.Fatalf("unexpected backend path: %s", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, chat.CompletionResponse{
			ID:      "chatcmpl-http-test",
			Object:  "chat.completion",
			Created: 456,
			Model:   "qwen2.5-coder-1.5b-q4",
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "served over http",
					},
					FinishReason: "stop",
				},
			},
			Usage: &chat.Usage{CompletionTokens: 3, TotalTokens: 9},
		})
	}))
	defer backend.Close()

	recorder := &recordingRecorder{}
	controlServer := httptest.NewServer(NewServer(
		"dev-token",
		testRegistry(),
		WithBenchmarkRecorder(recorder),
	).Router())
	defer controlServer.Close()

	registerNodeHTTP(t, controlServer.URL, cluster.HeartbeatRequest{
		NodeName: "jetson-01",
		Arch:     "arm64",
		OS:       "linux",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 8.0,
		},
		Backends: []cluster.RuntimeBackend{
			{
				ID:               cluster.BackendIDLlamaLocal,
				Kind:             cluster.RuntimeKindLlamaCPP,
				BaseURL:          backend.URL,
				Models:           []string{"qwen2.5-coder-1.5b-q4"},
				OpenAICompatible: true,
			},
		},
	})

	decoded := postChatHTTP(t, controlServer.URL, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello over HTTP."},
		},
	})
	choice := firstChoice(t, decoded)
	if choice.Message.Content != "served over http" {
		t.Fatalf("unexpected response content: %s", choice.Message.Content)
	}
	if decoded.Route == nil {
		t.Fatal("expected route metadata")
	}
	if decoded.Route.NodeName != "jetson-01" || decoded.Route.Mode != cluster.RouteModeSingleNode {
		t.Fatalf("unexpected route metadata: %+v", decoded.Route)
	}
	if len(recorder.records) != 1 {
		t.Fatalf("expected one benchmark record, got %d", len(recorder.records))
	}
}

func TestServerRunsOverHTTPThroughAgentProxy(t *testing.T) {
	runtimeBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathChatCompletions {
			t.Fatalf("unexpected runtime path: %s", r.URL.Path)
		}
		var req chat.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode runtime request: %v", err)
		}
		if req.Model != "qwen2.5-coder-1.5b-q4" {
			t.Fatalf("unexpected model routed to runtime: %s", req.Model)
		}
		writeJSON(w, http.StatusOK, chat.CompletionResponse{
			ID:      "chatcmpl-agent-proxy-test",
			Object:  "chat.completion",
			Created: 789,
			Model:   req.Model,
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "control routed through agent",
					},
					FinishReason: "stop",
				},
			},
			Usage: &chat.Usage{CompletionTokens: 4, TotalTokens: 11},
		})
	}))
	defer runtimeBackend.Close()

	runtimeClient, err := runtimeclient.NewOpenAIClient(runtimeBackend.URL, time.Second)
	if err != nil {
		t.Fatalf("create runtime client: %v", err)
	}
	agentServer := httptest.NewServer(agent.NewServer(runtimeClient).Router())
	defer agentServer.Close()

	recorder := &recordingRecorder{}
	controlServer := httptest.NewServer(NewServer(
		"dev-token",
		testRegistry(),
		WithBenchmarkRecorder(recorder),
	).Router())
	defer controlServer.Close()

	registerNodeHTTP(t, controlServer.URL, cluster.HeartbeatRequest{
		NodeName: "jetson-agent-01",
		Arch:     "arm64",
		OS:       "linux",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 8.0,
		},
		Backends: []cluster.RuntimeBackend{
			{
				ID:               cluster.BackendIDLlamaLocal,
				Kind:             cluster.RuntimeKindLlamaCPP,
				BaseURL:          agentServer.URL,
				Models:           []string{"qwen2.5-coder-1.5b-q4"},
				OpenAICompatible: true,
			},
		},
	})

	decoded := postChatHTTP(t, controlServer.URL, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Route through the agent."},
		},
	})
	choice := firstChoice(t, decoded)
	if choice.Message.Content != "control routed through agent" {
		t.Fatalf("unexpected response content: %s", choice.Message.Content)
	}
	if decoded.Route == nil {
		t.Fatal("expected route metadata")
	}
	if decoded.Route.NodeName != "jetson-agent-01" || decoded.Route.BackendID != cluster.BackendIDLlamaLocal {
		t.Fatalf("unexpected route metadata: %+v", decoded.Route)
	}
	if decoded.Route.BackendKind != cluster.RuntimeKindLlamaCPP {
		t.Fatalf("unexpected backend kind: %s", decoded.Route.BackendKind)
	}
	if len(recorder.records) != 1 {
		t.Fatalf("expected one benchmark record, got %d", len(recorder.records))
	}
	record := firstRecord(t, recorder.records)
	if record.NodeName != "jetson-agent-01" || record.OutputTokens != 4 {
		t.Fatalf("unexpected benchmark record: %+v", record)
	}
}

func TestLayerSplitPlanUsesRegisteredAgents(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-1", 1))
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-2", 1))
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-3", 1))

	request := httptest.NewRequest(http.MethodGet, api.PathLayerSplitPlan+"?model=qwen2.5-coder-1.5b-q4", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", response.Code, response.Body.String())
	}

	var decoded map[string]any
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode layer split plan: %v", err)
	}
	if decoded["mode"] != string(cluster.RouteModeLayerSplit) {
		t.Fatalf("unexpected mode: %+v", decoded)
	}
	stages, ok := decoded["stages"].([]any)
	if !ok || len(stages) != 3 {
		t.Fatalf("expected three stages, got %+v", decoded["stages"])
	}
	assertLayerRange(t, stages[0], "desktop-agent-1", 0, 10)
	assertLayerRange(t, stages[1], "desktop-agent-2", 10, 19)
	assertLayerRange(t, stages[2], "desktop-agent-3", 19, 28)
}

func TestLayerSplitPlanRequiresTwoCandidates(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-1", 1))

	request := httptest.NewRequest(http.MethodGet, api.PathLayerSplitPlan+"?model=qwen2.5-coder-1.5b-q4", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, errorNoLayerSplitRoute)
}

func TestLayerSplitCompletionsRunsAllStages(t *testing.T) {
	transport := &layersplit.LocalTransport{}
	recorder := &recordingRecorder{}
	handler := NewServer(
		"dev-token",
		testRegistry(),
		WithLayerTransport(layersplit.TransportLocal, transport),
		WithBenchmarkRecorder(recorder),
		WithClock(func() time.Time { return time.Unix(200, 0).UTC() }),
	).Router()
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-1", 1))
	registerNode(t, handler, layerSplitHeartbeat("desktop-agent-2", 1))

	body, err := json.Marshal(chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello from both agents."},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitChat, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", response.Code, response.Body.String())
	}

	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	choice := firstChoice(t, decoded)
	expected := "synthetic layer_split response: Say hello from both agents. -> desktop-agent-1[0:14] -> desktop-agent-2[14:28]"
	if choice.Message.Content != expected {
		t.Fatalf("unexpected content: %s", choice.Message.Content)
	}
	if decoded.Route == nil || decoded.Route.Mode != cluster.RouteModeLayerSplit {
		t.Fatalf("expected layer split route, got %+v", decoded.Route)
	}
	if len(decoded.Route.Stages) != 2 {
		t.Fatalf("expected two route stages, got %+v", decoded.Route.Stages)
	}
	if decoded.Route.Stages[0].NodeName != "desktop-agent-1" || decoded.Route.Stages[1].NodeName != "desktop-agent-2" {
		t.Fatalf("unexpected route stages: %+v", decoded.Route.Stages)
	}
	if len(transport.Requests) != 2 {
		t.Fatalf("expected two transport requests, got %d", len(transport.Requests))
	}
	if len(recorder.records) != 1 || recorder.records[0].RouteMode != cluster.RouteModeLayerSplit {
		t.Fatalf("expected one layer split benchmark record, got %+v", recorder.records)
	}
}

func TestChatCompletionsRejectsMissingBackend(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	registerNode(t, handler, cluster.HeartbeatRequest{
		NodeName: "jetson-01",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 8.0,
		},
	})

	response := postChat(t, handler, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected status 503, got %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, errorNoSingleNodeRoute)
}

func TestChatCompletionsRejectsUnknownModel(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	response := postChat(t, handler, chat.CompletionRequest{
		Model: "missing-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if response.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, errorUnknownModel)
}

func TestChatCompletionsRejectsInvalidBackendURL(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	registerNode(t, handler, cluster.HeartbeatRequest{
		NodeName: "jetson-01",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 8.0,
		},
		Backends: []cluster.RuntimeBackend{
			{
				ID:               cluster.BackendIDLlamaLocal,
				Kind:             cluster.RuntimeKindLlamaCPP,
				BaseURL:          "127.0.0.1:8080",
				Models:           []string{"qwen2.5-coder-1.5b-q4"},
				OpenAICompatible: true,
			},
		},
	})

	response := postChat(t, handler, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	if response.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, errorBackendConfigInvalid)
}

func testRegistry() modelregistry.Registry {
	return modelregistry.Registry{
		Models: []cluster.ModelProfile{
			{
				ID:             "qwen2.5-coder-1.5b-q4",
				Family:         "llm",
				Runtime:        cluster.RuntimeKindLlamaCPP,
				LayerCount:     28,
				MinMemoryGB:    3,
				PlacementModes: []cluster.RouteMode{cluster.RouteModeSingleNode, cluster.RouteModeLayerSplit},
			},
		},
	}
}

func layerSplitHeartbeat(nodeName string, weight float64) cluster.HeartbeatRequest {
	return cluster.HeartbeatRequest{
		NodeName: nodeName,
		Arch:     "amd64",
		OS:       "linux",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:     16.0,
			cluster.CapabilityLayerWeight:  weight,
			cluster.CapabilityAccelerators: []any{cluster.AcceleratorCUDA},
		},
		Backends: []cluster.RuntimeBackend{
			{
				ID:               cluster.BackendIDLlamaLocal,
				Kind:             cluster.RuntimeKindLlamaCPP,
				BaseURL:          "http://127.0.0.1:52416",
				Models:           []string{"qwen2.5-coder-1.5b-q4"},
				OpenAICompatible: true,
			},
		},
	}
}

func registerNode(t *testing.T, handler http.Handler, heartbeat cluster.HeartbeatRequest) {
	t.Helper()
	body, err := json.Marshal(heartbeat)
	if err != nil {
		t.Fatalf("marshal heartbeat: %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathAgentHeartbeat, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer dev-token")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("heartbeat failed with %d: %s", response.Code, response.Body.String())
	}
}

func postChat(t *testing.T, handler http.Handler, requestBody chat.CompletionRequest) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(requestBody)
	if err != nil {
		t.Fatalf("marshal chat request: %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func registerNodeHTTP(t *testing.T, baseURL string, heartbeat cluster.HeartbeatRequest) {
	t.Helper()
	url := joinedURL(t, baseURL, api.PathAgentHeartbeat)
	response := postJSON(t, url, heartbeat, func(request *http.Request) {
		request.Header.Set("Authorization", "Bearer dev-token")
	})
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatalf("read heartbeat error response: %v", err)
		}
		t.Fatalf("heartbeat failed with %d: %s", response.StatusCode, string(body))
	}
}

func postChatHTTP(t *testing.T, baseURL string, requestBody chat.CompletionRequest) chat.CompletionResponse {
	t.Helper()
	url := joinedURL(t, baseURL, api.PathChatCompletions)
	response := postJSON(t, url, requestBody, nil)
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatalf("read chat error response: %v", err)
		}
		t.Fatalf("chat request failed with %d: %s", response.StatusCode, string(body))
	}
	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode chat response: %v", err)
	}
	return decoded
}

func postJSON(t *testing.T, url string, payload any, configure func(*http.Request)) *http.Response {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	request, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if configure != nil {
		configure(request)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("send request: %v", err)
	}
	return response
}

func joinedURL(t *testing.T, baseURL string, path string) string {
	t.Helper()
	url, err := api.JoinBasePath(baseURL, path)
	if err != nil {
		t.Fatalf("join URL: %v", err)
	}
	return url
}

func assertErrorCode(t *testing.T, response *httptest.ResponseRecorder, expected errorCode) {
	t.Helper()
	var payload map[string]string
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if payload["error"] != string(expected) {
		t.Fatalf("expected error %q, got %q", expected, payload["error"])
	}
}

func firstChoice(t *testing.T, response chat.CompletionResponse) chat.Choice {
	t.Helper()
	for _, choice := range response.Choices {
		return choice
	}
	t.Fatal("expected at least one choice")
	return chat.Choice{}
}

func firstRecord(t *testing.T, records []benchmarks.Record) benchmarks.Record {
	t.Helper()
	for _, record := range records {
		return record
	}
	t.Fatal("expected at least one benchmark record")
	return benchmarks.Record{}
}

func assertLayerRange(t *testing.T, stage any, nodeName string, start int, end int) {
	t.Helper()
	stageMap, ok := stage.(map[string]any)
	if !ok {
		t.Fatalf("expected stage object, got %+v", stage)
	}
	if stageMap["node_name"] != nodeName {
		t.Fatalf("expected node %s, got %+v", nodeName, stageMap)
	}
	if int(stageMap["layer_start"].(float64)) != start || int(stageMap["layer_end"].(float64)) != end {
		t.Fatalf("unexpected layer range: %+v", stageMap)
	}
}

type recordingRecorder struct {
	records []benchmarks.Record
}

func (r *recordingRecorder) Record(_ context.Context, record benchmarks.Record) error {
	r.records = append(r.records, record)
	return nil
}
