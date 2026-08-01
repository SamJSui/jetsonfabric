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
			Trace: &chat.RuntimeTrace{
				StageCalls:                1,
				TargetDecodePasses:        1,
				SpeculativeDraftTokens:    2,
				SpeculativeAcceptedTokens: 1,
				StageTimings: []chat.StageTiming{{
					Phase: "decode", StageIndex: 0, NodeName: "desktop-agent-1",
					Calls: 2, BatchItems: 4, MaxExecutionBatch: 2,
					ExecutionUS: 400, StageTotalUS: 500,
				}},
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
	if summary.MeanOutputTokens != 2 ||
		summary.RequestThroughput <= 0 ||
		summary.OutputThroughput <= 0 ||
		summary.TokensPerSecond != summary.OutputThroughput {
		t.Fatalf("unexpected serving throughput: %+v", summary)
	}
	if summary.Results[0].Route == nil || summary.Results[0].Route.NodeName != "desktop-agent-1" {
		t.Fatalf("expected route metadata, got %+v", summary.Results[0])
	}
	if summary.Runtime.StageCalls != 3 || len(summary.Runtime.StageTimings) != 1 ||
		summary.Runtime.StageTimings[0].Calls != 6 ||
		summary.Runtime.StageTimings[0].Execution.AvgUS != 200 ||
		summary.Runtime.StageTimings[0].BatchItems != 12 ||
		summary.Runtime.StageTimings[0].MeanBatchSize != 2 ||
		summary.Runtime.StageTimings[0].MaxBatchSize != 2 {
		t.Fatalf("runtime timings were not summarized: %+v", summary.Runtime)
	}
	if summary.Runtime.TargetDecodePasses != 3 ||
		summary.Runtime.SpeculativeDraftTokens != 6 ||
		summary.Runtime.SpeculativeAcceptedTokens != 3 ||
		summary.Runtime.SpeculativeAcceptanceRate != 0.5 {
		t.Fatalf("speculative telemetry was not summarized: %+v", summary.Runtime)
	}
}

func TestRunBenchmarkEndpointsDistributesReplicaRequests(t *testing.T) {
	counts := make([]int, 2)
	servers := make([]*httptest.Server, 2)
	for index := range servers {
		serverIndex := index
		servers[index] = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			counts[serverIndex]++
			writeJSON(w, http.StatusOK, chat.CompletionResponse{
				Model: "model-a",
				Usage: &chat.Usage{CompletionTokens: 1},
			})
		}))
		defer servers[index].Close()
	}
	endpoints := []string{
		servers[0].URL + api.PathChatCompletions,
		servers[1].URL + api.PathChatCompletions,
	}
	summary, err := runBenchmarkEndpoints(
		t.Context(),
		http.DefaultClient,
		endpoints,
		benchmarkRequest{Body: []byte(`{"model":"model-a","messages":[{"role":"user","content":"hi"}]}`)},
		4,
		2,
		1,
		time.Second,
	)
	if err != nil {
		t.Fatal(err)
	}
	if counts[0] != 3 || counts[1] != 3 {
		t.Fatalf("requests were not evenly distributed: %v", counts)
	}
	if summary.Endpoint != "" || len(summary.Endpoints) != 2 ||
		summary.Results[0].Endpoint != endpoints[0] ||
		summary.Results[1].Endpoint != endpoints[1] {
		t.Fatalf("replica endpoints were not recorded: %+v", summary)
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

func TestConsumeSSEMeasuresRuntimeTokensIncludingEmptyText(t *testing.T) {
	body := strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"jetsonfabric\":{\"token_index\":0}}\n\n" +
			"data: {\"choices\":[{\"delta\":{}}],\"jetsonfabric\":{\"token_index\":1}}\n\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}],\"jetsonfabric\":{\"token_index\":2}}\n\n" +
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"completion_tokens\":3},\"jetsonfabric\":{\"stage_calls\":3,\"stage_timings\":[{\"phase\":\"decode\",\"stage_index\":0,\"node_name\":\"node-a\",\"calls\":3,\"execution_us\":30,\"stage_total_us\":36}]}}\n\n" +
			"data: [DONE]\n\n",
	)
	startedAt := time.Unix(100, 0)
	times := []time.Time{
		startedAt.Add(10 * time.Millisecond),
		startedAt.Add(17 * time.Millisecond),
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
	if metrics.TTFTMS != 10 || metrics.OutputTokens != 3 ||
		len(metrics.InterTokenLatencyMS) != 2 ||
		metrics.InterTokenLatencyMS[0] != 7 ||
		metrics.InterTokenLatencyMS[1] != 8 {
		t.Fatalf("unexpected stream metrics: %+v", metrics)
	}
	if metrics.Trace == nil || metrics.Trace.StageCalls != 3 ||
		len(metrics.Trace.StageTimings) != 1 {
		t.Fatalf("runtime trace was not captured: %+v", metrics.Trace)
	}
}

func TestConsumeSSERetainsPartialMetricsOnStreamError(t *testing.T) {
	body := strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}],\"jetsonfabric\":{\"token_index\":0}}\n\n" +
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
		summary.P50MS != 3 || summary.P90MS != 5 ||
		summary.P95MS != 5 || summary.P99MS != 5 {
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
