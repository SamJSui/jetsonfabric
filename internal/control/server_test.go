package control

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/api"
	"github.com/SamJSui/JetsonMesh/internal/benchmarks"
	"github.com/SamJSui/JetsonMesh/internal/chat"
	"github.com/SamJSui/JetsonMesh/internal/cluster"
	"github.com/SamJSui/JetsonMesh/internal/modelregistry"
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
		NodeID: "jetson-01",
		Arch:   "arm64",
		OS:     "linux",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:     8.0,
			cluster.CapabilityAccelerators: []any{cluster.AcceleratorJetson, cluster.AcceleratorCUDA},
		},
		Metrics: map[string]any{
			cluster.MetricTemperatureC: 42.5,
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
	if decoded.Route.Mode != cluster.RouteModeSingleNode || decoded.Route.NodeID != "jetson-01" || decoded.Route.BackendID != cluster.BackendIDLlamaLocal {
		t.Fatalf("unexpected route metadata: %+v", decoded.Route)
	}
	if len(recorder.records) != 1 {
		t.Fatalf("expected one benchmark record, got %d", len(recorder.records))
	}
	record := firstRecord(t, recorder.records)
	if record.ModelID != "qwen2.5-coder-1.5b-q4" || record.NodeID != "jetson-01" || record.RouteMode != cluster.RouteModeSingleNode {
		t.Fatalf("unexpected benchmark record: %+v", record)
	}
	if record.MemoryGB == nil || *record.MemoryGB != 8.0 {
		t.Fatalf("expected memory benchmark field, got %+v", record.MemoryGB)
	}
	if record.TemperatureC == nil || *record.TemperatureC != 42.5 {
		t.Fatalf("expected temperature benchmark field, got %+v", record.TemperatureC)
	}
}

func TestChatCompletionsRejectsMissingBackend(t *testing.T) {
	handler := NewServer("dev-token", testRegistry()).Router()
	registerNode(t, handler, cluster.HeartbeatRequest{
		NodeID: "jetson-01",
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
		NodeID: "jetson-01",
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
				MinMemoryGB:    3,
				PlacementModes: []cluster.RouteMode{cluster.RouteModeSingleNode},
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

type recordingRecorder struct {
	records []benchmarks.Record
}

func (r *recordingRecorder) Record(_ context.Context, record benchmarks.Record) error {
	r.records = append(r.records, record)
	return nil
}
