package runtimegateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

func TestChatProxyReturnsUsageFromRuntimeTokenCounts(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected runtime path: %s", r.URL.Path)
		}
		stageReq := decodeStageRequest(t, r)
		if stageReq.MaxTokens != 32 {
			t.Fatalf("expected max_tokens=32, got %d", stageReq.MaxTokens)
		}
		if !strings.Contains(stageReq.Payload, "<|im_start|>user\nSay hello") {
			t.Fatalf("expected rendered Qwen prompt, got %q", stageReq.Payload)
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"payload":           " Hello from test. ",
			"bytes_in":          stageReq.BytesIn,
			"bytes_out":         16,
			"prompt_tokens":     11,
			"completion_tokens": 4,
		})
	}))
	defer runtime.Close()

	proxy, err := NewChatProxy(ChatProxyConfig{RuntimeURL: runtime.URL, NodeName: "dopey", Model: "qwen"})
	if err != nil {
		t.Fatalf("create proxy: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":32
	}`))
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("expected OK, got %d: %s", response.Code, response.Body.String())
	}
	var decoded chatCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got := decoded.Choices[0].Message.Content; got != "Hello from test." {
		t.Fatalf("expected trimmed assistant content, got %q", got)
	}
	if decoded.Usage.PromptTokens != 11 || decoded.Usage.CompletionTokens != 4 || decoded.Usage.TotalTokens != 15 {
		t.Fatalf("unexpected usage: %+v", decoded.Usage)
	}
}

func decodeStageRequest(t *testing.T, r *http.Request) stageRequest {
	t.Helper()
	var stageReq stageRequest
	if err := json.NewDecoder(r.Body).Decode(&stageReq); err != nil {
		t.Fatalf("decode stage request: %v", err)
	}
	return stageReq
}
