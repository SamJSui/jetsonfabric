package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestRunBenchmarkRecordsSuccessfulRequests(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathChatCompletions {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, chat.CompletionResponse{
			ID:      "chatcmpl-bench-test",
			Object:  "chat.completion",
			Created: 123,
			Model:   "qwen2.5-coder-1.5b-q4",
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "bench response",
					},
					FinishReason: "stop",
				},
			},
			Usage: &chat.Usage{CompletionTokens: 2, TotalTokens: 8},
			Route: &chat.RouteMetadata{
				Mode:        cluster.RouteModeSingleNode,
				NodeName:    "desktop-agent-1",
				BackendID:   cluster.BackendIDLlamaLocal,
				BackendKind: cluster.RuntimeKindLlamaCPP,
				LatencyMS:   3,
			},
		})
	}))
	defer server.Close()

	summary, err := runBenchmark(
		t.Context(),
		server.Client(),
		server.URL+api.PathChatCompletions,
		chat.CompletionRequest{
			Model: "qwen2.5-coder-1.5b-q4",
			Messages: []chat.Message{
				{Role: "user", Content: "hello"},
			},
		},
		3,
		2,
		time.Second,
	)
	if err != nil {
		t.Fatalf("run benchmark: %v", err)
	}
	if summary.SuccessCount != 3 || summary.FailureCount != 0 {
		t.Fatalf("unexpected success/failure counts: %+v", summary)
	}
	if summary.OutputTokens != 6 {
		t.Fatalf("unexpected output tokens: %d", summary.OutputTokens)
	}
	if summary.Results[0].Route == nil || summary.Results[0].Route.NodeName != "desktop-agent-1" {
		t.Fatalf("expected route metadata, got %+v", summary.Results[0])
	}
}

func TestRunBenchmarkRejectsInvalidCount(t *testing.T) {
	_, err := runBenchmark(t.Context(), nil, "http://example.invalid", chat.CompletionRequest{}, 0, 1, time.Second)
	if err == nil {
		t.Fatal("expected invalid count to fail")
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		panic(err)
	}
}
