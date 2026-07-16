package runtimebridge

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestChatProxyUsesBinaryStagewireAndReturnsUsage(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected runtime path: %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != stagewire.ContentType {
			t.Fatalf("content-type=%q", r.Header.Get("Content-Type"))
		}
		stageReq, err := stagewire.Decode(r.Body)
		if err != nil {
			t.Fatalf("decode stage request: %v", err)
		}
		if stageReq.MaxTokens != 32 || !strings.Contains(string(stageReq.Payload), "<|im_start|>user\nSay hello") {
			t.Fatalf("unexpected request: %+v payload=%q", stageReq.Metadata, stageReq.Payload)
		}
		metadata := stageReq.Metadata
		metadata.PayloadKind = stagewire.PayloadKindText
		metadata.Encoding = "utf-8"
		metadata.PromptTokens = 11
		metadata.CompletionTokens = 4
		metadata.BytesIn = int64(len(stageReq.Payload))
		metadata.BytesOut = 16
		encoded, err := stagewire.Marshal(stagewire.StageResponse{Metadata: metadata, Payload: []byte(" Hello from test. ")})
		if err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", stagewire.ContentType)
		_, _ = w.Write(encoded)
	}))
	defer runtime.Close()

	proxy, err := NewChatProxy(ChatProxyConfig{RuntimeURL: runtime.URL, NodeName: "dopey", Model: "qwen"})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"model":"qwen",
		"messages":[{"role":"user","content":"Say hello"}],
		"max_tokens":32
	}`))
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var decoded chatCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Choices[0].Message.Content != "Hello from test." || decoded.Usage.TotalTokens != 15 {
		t.Fatalf("unexpected response: %+v", decoded)
	}
}

func TestChatProxyRejectsStreaming(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("runtime should not be called")
	}))
	defer runtime.Close()
	proxy, _ := NewChatProxy(ChatProxyConfig{RuntimeURL: runtime.URL})
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{
		"messages":[{"role":"user","content":"hello"}],"stream":true
	}`))
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", response.Code)
	}
}

func TestChatProxyRejectsEmptyMessages(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("runtime should not be called")
	}))
	defer runtime.Close()
	proxy, _ := NewChatProxy(ChatProxyConfig{RuntimeURL: runtime.URL})
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{"messages":[{"role":"user","content":" "}]}`))
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", response.Code)
	}
}

func TestChatProxyReturnsRuntimeJSONError(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime_down", "message": "unavailable"})
	}))
	defer runtime.Close()
	proxy, _ := NewChatProxy(ChatProxyConfig{RuntimeURL: runtime.URL})
	request := httptest.NewRequest(http.MethodPost, api.PathChatCompletions, strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "error", "runtime_down")
}
