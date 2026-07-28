package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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
				Mode:             cluster.ExecutionModeDataParallel,
				NodeName:         "desktop-agent-1",
				Engine:           cluster.EngineLlamaCPP,
				EngineInstanceID: cluster.DefaultEngineInstanceID,
				LatencyMS:        3,
			},
		})
	}))
	defer server.Close()

	summary, err := runBenchmark(
		t.Context(),
		server.Client(),
		server.URL+api.PathChatCompletions,
		benchmarkRequest{Body: []byte(`{
			"model":"qwen2.5-coder-1.5b-q4",
			"messages":[{"role":"user","content":"hello"}],
			"max_tokens":2
		}`)},
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
	_, err := runBenchmark(t.Context(), nil, "http://example.invalid", benchmarkRequest{}, 0, 1, time.Second)
	if err == nil {
		t.Fatal("expected invalid count to fail")
	}
}

func TestEnableStreamingPreservesRequestFields(t *testing.T) {
	request := benchmarkRequest{Body: []byte(`{
		"model":"qwen2.5-coder-1.5b-q4",
		"messages":[{"role":"user","content":"hello"}],
		"max_tokens":64,
		"jetsonfabric":{"stage_count":2}
	}`)}
	streaming, err := enableStreaming(request)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(streaming.Body, &decoded); err != nil {
		t.Fatal(err)
	}
	if !streaming.Streaming || string(decoded["stream"]) != "true" ||
		string(decoded["max_tokens"]) != "64" ||
		!bytes.Contains(decoded["jetsonfabric"], []byte(`"stage_count":2`)) {
		t.Fatalf("request fields were not preserved: %s", streaming.Body)
	}
}

func TestConsumeSSEMeasuresFirstContentAndInterTokenLatency(t *testing.T) {
	body := strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n" +
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	startedAt := time.Unix(100, 0)
	times := []time.Time{
		startedAt.Add(10 * time.Millisecond),
		startedAt.Add(25 * time.Millisecond),
	}
	next := 0
	metrics, err := consumeSSE(body, startedAt, func() time.Time {
		value := times[next]
		next++
		return value
	})
	if err != nil {
		t.Fatal(err)
	}
	if metrics.TTFTMS != 10 || metrics.OutputTokens != 2 ||
		len(metrics.InterTokenLatencyMS) != 1 || metrics.InterTokenLatencyMS[0] != 15 {
		t.Fatalf("unexpected stream metrics: %+v", metrics)
	}
}

func TestConsumeSSERetainsPartialMetricsOnStreamError(t *testing.T) {
	body := strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
			"data: {\"error\":{\"code\":\"runtime_stage_unreachable\",\"message\":\"stage 1 failed\"}}\n\n",
	)
	startedAt := time.Unix(100, 0)
	metrics, err := consumeSSE(body, startedAt, func() time.Time {
		return startedAt.Add(12 * time.Millisecond)
	})
	if err == nil || !strings.Contains(err.Error(), "runtime_stage_unreachable") {
		t.Fatalf("expected stream error, got %v", err)
	}
	if metrics.TTFTMS != 12 || metrics.OutputTokens != 1 {
		t.Fatalf("partial stream metrics were lost: %+v", metrics)
	}
}

func TestSummarizeLatenciesUsesNearestRankPercentiles(t *testing.T) {
	summary := summarizeLatencies([]int64{5, 1, 3, 4, 2})
	if summary.MinMS != 1 || summary.MaxMS != 5 || summary.AvgMS != 3 ||
		summary.P50MS != 3 || summary.P95MS != 5 {
		t.Fatalf("unexpected latency summary: %+v", summary)
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		panic(err)
	}
}
