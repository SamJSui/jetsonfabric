package agent

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

func TestServerProxiesChatCompletionsToRuntimeBackend(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathChatCompletions {
			t.Fatalf("unexpected backend path: %s", r.URL.Path)
		}
		var req chat.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode backend request: %v", err)
		}
		if req.Model != "qwen2.5-coder-1.5b-q4" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		writeJSON(w, http.StatusOK, chat.CompletionResponse{
			ID:     "chatcmpl-agent-test",
			Object: "chat.completion",
			Model:  req.Model,
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "proxied by agent",
					},
					FinishReason: "stop",
				},
			},
			Usage: &chat.Usage{CompletionTokens: 3, TotalTokens: 8},
			Route: &chat.RouteMetadata{
				Mode:      cluster.RouteModeSingleNode,
				NodeID:    "runtime-should-not-set-control-route",
				LatencyMS: 1,
			},
		})
	}))
	defer backend.Close()

	runtimeBackend, err := runtimeclient.NewOpenAIClient(backend.URL, time.Second)
	if err != nil {
		t.Fatalf("create runtime client: %v", err)
	}
	server := httptest.NewServer(NewServer(runtimeBackend).Router())
	defer server.Close()

	response := postChat(t, server.URL, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body := readBody(t, response)
		t.Fatalf("expected status 200, got %d: %s", response.StatusCode, body)
	}
	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode agent response: %v", err)
	}
	choice := firstChoice(t, decoded)
	if choice.Message.Content != "proxied by agent" {
		t.Fatalf("unexpected response content: %s", choice.Message.Content)
	}
	if decoded.Route != nil {
		t.Fatalf("agent should not return control-plane route metadata: %+v", decoded.Route)
	}
}

func TestServerRejectsChatWhenRuntimeBackendIsUnavailable(t *testing.T) {
	server := httptest.NewServer(NewServer(nil).Router())
	defer server.Close()

	response := postChat(t, server.URL, chat.CompletionRequest{
		Model: "qwen2.5-coder-1.5b-q4",
		Messages: []chat.Message{
			{Role: "user", Content: "Say hello."},
		},
	})
	defer response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable {
		body := readBody(t, response)
		t.Fatalf("expected status 503, got %d: %s", response.StatusCode, body)
	}
}

func TestServerRunsSyntheticLayerSplitStage(t *testing.T) {
	server := httptest.NewServer(NewServer(nil, WithNodeID("agent-stage-1")).Router())
	defer server.Close()

	payload := map[string]any{
		"session_id":  "session-1",
		"request_id":  "request-1",
		"model_id":    "qwen2.5-coder-1.5b-q4",
		"stage_index": 0,
		"node_id":     "agent-stage-1",
		"role":        "first",
		"layer_start": 0,
		"layer_end":   14,
		"payload":     "Say hello.",
		"bytes_in":    10,
		"transport":   "http",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	request, err := http.NewRequest(http.MethodPost, server.URL+api.PathLayerSplitStage, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("send request: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body := readBody(t, response)
		t.Fatalf("expected status 200, got %d: %s", response.StatusCode, body)
	}
	var decoded map[string]any
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if decoded["payload"] != "Say hello. -> agent-stage-1[0:14]" {
		t.Fatalf("unexpected payload: %+v", decoded)
	}
}

func postChat(t *testing.T, baseURL string, payload chat.CompletionRequest) *http.Response {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal chat request: %v", err)
	}
	request, err := http.NewRequest(http.MethodPost, baseURL+api.PathChatCompletions, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("send request: %v", err)
	}
	return response
}

func readBody(t *testing.T, response *http.Response) string {
	t.Helper()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return string(body)
}

func firstChoice(t *testing.T, response chat.CompletionResponse) chat.Choice {
	t.Helper()
	for _, choice := range response.Choices {
		return choice
	}
	t.Fatal("expected at least one choice")
	return chat.Choice{}
}
