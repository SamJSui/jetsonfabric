package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
)

const (
	defaultEndpoint    = "http://127.0.0.1:52415" + api.PathChatCompletions
	defaultRequestPath = "examples/chat-request.json"
	defaultBenchCount  = 1
	defaultConcurrency = 1
	outputFilePerm     = 0o644
	outputDirPerm      = 0o755
	maxErrorBodyBytes  = 4096
)

type Summary struct {
	Name              string          `json:"name,omitempty"`
	Endpoint          string          `json:"endpoint,omitempty"`
	Endpoints         []string        `json:"endpoints"`
	WarmupCount       int             `json:"warmup_count"`
	RequestSHA256     string          `json:"request_sha256"`
	Metadata          map[string]any  `json:"metadata,omitempty"`
	Streaming         bool            `json:"streaming"`
	RequestCount      int             `json:"request_count"`
	Concurrency       int             `json:"concurrency"`
	SuccessCount      int             `json:"success_count"`
	FailureCount      int             `json:"failure_count"`
	StartedAt         time.Time       `json:"started_at"`
	FinishedAt        time.Time       `json:"finished_at"`
	TotalDurationMS   int64           `json:"total_duration_ms"`
	Latency           LatencySummary  `json:"latency"`
	TTFT              LatencySummary  `json:"ttft,omitempty"`
	InterTokenLatency LatencySummary  `json:"inter_token_latency,omitempty"`
	OutputTokens      int             `json:"output_tokens"`
	MeanOutputTokens  float64         `json:"mean_output_tokens"`
	RequestThroughput float64         `json:"request_throughput"`
	OutputThroughput  float64         `json:"output_token_throughput"`
	TokensPerSecond   float64         `json:"tokens_per_second"`
	Runtime           RuntimeSummary  `json:"runtime"`
	Results           []RequestResult `json:"results"`
}

type LatencySummary struct {
	MinMS int64   `json:"min_ms"`
	MaxMS int64   `json:"max_ms"`
	AvgMS float64 `json:"avg_ms"`
	P50MS int64   `json:"p50_ms"`
	P90MS int64   `json:"p90_ms"`
	P95MS int64   `json:"p95_ms"`
	P99MS int64   `json:"p99_ms"`
}

type RequestResult struct {
	Index               int                 `json:"index"`
	Endpoint            string              `json:"endpoint"`
	StatusCode          int                 `json:"status_code,omitempty"`
	LatencyMS           int64               `json:"latency_ms"`
	TTFTMS              int64               `json:"ttft_ms,omitempty"`
	InterTokenLatencyMS []int64             `json:"inter_token_latency_ms,omitempty"`
	OutputTokens        int                 `json:"output_tokens,omitempty"`
	Route               *chat.RouteMetadata `json:"jetsonfabric_route,omitempty"`
	Trace               *chat.RuntimeTrace  `json:"jetsonfabric,omitempty"`
	Error               string              `json:"error,omitempty"`
}

type RuntimeSummary struct {
	StageCalls       int                  `json:"stage_calls"`
	RemoteStageCalls int                  `json:"remote_stage_calls"`
	BytesIn          int64                `json:"bytes_in"`
	BytesOut         int64                `json:"bytes_out"`
	StageTimings     []StageTimingSummary `json:"stage_timings,omitempty"`
}

type StageTimingSummary struct {
	Phase            string           `json:"phase"`
	StageIndex       int              `json:"stage_index"`
	NodeName         string           `json:"node_name"`
	Remote           bool             `json:"remote"`
	Calls            int              `json:"calls"`
	Execution        MicrosecondTotal `json:"execution"`
	ActivationDecode MicrosecondTotal `json:"activation_decode"`
	ActivationEncode MicrosecondTotal `json:"activation_encode"`
	StageTotal       MicrosecondTotal `json:"stage_total"`
	RemoteCall       MicrosecondTotal `json:"remote_call"`
	RemoteOverhead   MicrosecondTotal `json:"remote_overhead"`
	BytesIn          int64            `json:"bytes_in"`
	BytesOut         int64            `json:"bytes_out"`
}

type MicrosecondTotal struct {
	TotalUS int64   `json:"total_us"`
	AvgUS   float64 `json:"avg_us"`
}

type benchmarkRequest struct {
	Body      []byte
	Streaming bool
}

type completionRequestEnvelope struct {
	Model    string            `json:"model"`
	Messages []json.RawMessage `json:"messages"`
	Stream   bool              `json:"stream,omitempty"`
}

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Content *string `json:"content,omitempty"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *chat.Usage `json:"usage,omitempty"`
	Trace *struct {
		TokenIndex       *int               `json:"token_index,omitempty"`
		StageCalls       int                `json:"stage_calls,omitempty"`
		RemoteStageCalls int                `json:"remote_stage_calls,omitempty"`
		BytesIn          int64              `json:"bytes_in,omitempty"`
		BytesOut         int64              `json:"bytes_out,omitempty"`
		StageTimings     []chat.StageTiming `json:"stage_timings,omitempty"`
	} `json:"jetsonfabric,omitempty"`
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

type streamMetrics struct {
	TTFTMS              int64
	InterTokenLatencyMS []int64
	OutputTokens        int
	Trace               *chat.RuntimeTrace
}

func main() {
	endpoint := flag.String("url", defaultEndpoint, "chat completions endpoint URL")
	endpointsCSV := flag.String("urls", "", "comma-separated chat endpoints for replica load distribution")
	requestPath := flag.String("request", defaultRequestPath, "chat request JSON path")
	count := flag.Int("count", defaultBenchCount, "number of requests to send")
	warmup := flag.Int("warmup", 0, "number of unmeasured warmup requests")
	concurrency := flag.Int("concurrency", defaultConcurrency, "number of concurrent workers")
	timeout := flag.Duration("timeout", 2*time.Minute, "per-request timeout")
	stream := flag.Bool("stream", false, "force stream=true and report TTFT/inter-token latency")
	manifestPath := flag.String("manifest", "", "optional benchmark suite manifest path")
	outputPath := flag.String("output", "", "optional JSON output path")
	flag.Parse()

	if strings.TrimSpace(*manifestPath) != "" {
		report, err := runManifest(context.Background(), http.DefaultClient, *manifestPath)
		if err != nil {
			log.Fatalf("run manifest: %v", err)
		}
		if err := writeJSONOutput(report, *outputPath); err != nil {
			log.Fatalf("write report: %v", err)
		}
		return
	}

	request, err := loadCompletionRequest(*requestPath)
	if err != nil {
		log.Fatalf("load request: %v", err)
	}
	if *stream && !request.Streaming {
		request, err = enableStreaming(request)
		if err != nil {
			log.Fatalf("enable streaming: %v", err)
		}
	}
	endpoints, err := benchmarkEndpoints(*endpoint, *endpointsCSV)
	if err != nil {
		log.Fatalf("endpoints: %v", err)
	}
	summary, err := runBenchmarkEndpoints(
		context.Background(),
		http.DefaultClient,
		endpoints,
		request,
		*count,
		*warmup,
		*concurrency,
		*timeout,
	)
	if err != nil {
		log.Fatalf("run benchmark: %v", err)
	}
	if err := writeJSONOutput(summary, *outputPath); err != nil {
		log.Fatalf("write summary: %v", err)
	}
}

func loadCompletionRequest(path string) (benchmarkRequest, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return benchmarkRequest{}, err
	}
	var envelope completionRequestEnvelope
	if err := json.Unmarshal(content, &envelope); err != nil {
		return benchmarkRequest{}, err
	}
	if strings.TrimSpace(envelope.Model) == "" {
		return benchmarkRequest{}, fmt.Errorf("request model is required")
	}
	if len(envelope.Messages) == 0 {
		return benchmarkRequest{}, fmt.Errorf("request messages are required")
	}
	return benchmarkRequest{Body: bytes.TrimSpace(content), Streaming: envelope.Stream}, nil
}

func enableStreaming(request benchmarkRequest) (benchmarkRequest, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(request.Body, &object); err != nil {
		return benchmarkRequest{}, err
	}
	object["stream"] = json.RawMessage("true")
	body, err := json.Marshal(object)
	if err != nil {
		return benchmarkRequest{}, err
	}
	return benchmarkRequest{Body: body, Streaming: true}, nil
}

func benchmarkEndpoints(single string, csv string) ([]string, error) {
	values := []string{single}
	if strings.TrimSpace(csv) != "" {
		values = strings.Split(csv, ",")
	}
	endpoints := make([]string, 0, len(values))
	for _, value := range values {
		endpoint := strings.TrimSpace(value)
		if endpoint == "" {
			return nil, fmt.Errorf("endpoint must not be empty")
		}
		endpoints = append(endpoints, endpoint)
	}
	if len(endpoints) == 0 {
		return nil, fmt.Errorf("at least one endpoint is required")
	}
	return endpoints, nil
}

func runBenchmark(
	ctx context.Context,
	client *http.Client,
	endpoint string,
	request benchmarkRequest,
	count int,
	concurrency int,
	timeout time.Duration,
) (Summary, error) {
	return runBenchmarkEndpoints(
		ctx,
		client,
		[]string{endpoint},
		request,
		count,
		0,
		concurrency,
		timeout,
	)
}

func runBenchmarkEndpoints(
	ctx context.Context,
	client *http.Client,
	endpoints []string,
	request benchmarkRequest,
	count int,
	warmup int,
	concurrency int,
	timeout time.Duration,
) (Summary, error) {
	if count <= 0 {
		return Summary{}, fmt.Errorf("count must be greater than zero")
	}
	if warmup < 0 {
		return Summary{}, fmt.Errorf("warmup must not be negative")
	}
	if len(endpoints) == 0 {
		return Summary{}, fmt.Errorf("at least one endpoint is required")
	}
	if concurrency <= 0 {
		return Summary{}, fmt.Errorf("concurrency must be greater than zero")
	}
	if concurrency > count {
		concurrency = count
	}
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	if client == nil {
		client = http.DefaultClient
	}

	if warmup > 0 {
		warmupResults := executeRequests(ctx, client, endpoints, request, warmup, concurrency, timeout)
		for _, result := range warmupResults {
			if result.Error != "" {
				return Summary{}, fmt.Errorf("warmup request %d failed: %s", result.Index, result.Error)
			}
		}
	}

	startedAt := time.Now().UTC()
	results := executeRequests(ctx, client, endpoints, request, count, concurrency, timeout)
	finishedAt := time.Now().UTC()

	summary := summarize(endpoints[0], request.Streaming, count, concurrency, startedAt, finishedAt, results)
	summary.Endpoints = append([]string(nil), endpoints...)
	summary.WarmupCount = warmup
	summary.RequestSHA256 = fmt.Sprintf("%x", sha256.Sum256(request.Body))
	if len(endpoints) != 1 {
		summary.Endpoint = ""
	}
	return summary, nil
}

func executeRequests(
	ctx context.Context,
	client *http.Client,
	endpoints []string,
	request benchmarkRequest,
	count int,
	concurrency int,
	timeout time.Duration,
) []RequestResult {
	if concurrency > count {
		concurrency = count
	}
	results := make([]RequestResult, count)
	jobs := make(chan int)
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				requestCtx, cancel := context.WithTimeout(ctx, timeout)
				endpoint := endpoints[index%len(endpoints)]
				results[index] = sendRequest(requestCtx, client, endpoint, request, index)
				cancel()
			}
		}()
	}
	for index := 0; index < count; index++ {
		jobs <- index
	}
	close(jobs)
	wg.Wait()
	return results
}

func sendRequest(ctx context.Context, client *http.Client, endpoint string, request benchmarkRequest, index int) RequestResult {
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(request.Body))
	if err != nil {
		return RequestResult{Index: index, Endpoint: endpoint, Error: fmt.Sprintf("create request: %v", err)}
	}
	httpRequest.Header.Set("Content-Type", "application/json")

	start := time.Now()
	response, err := client.Do(httpRequest)
	if err != nil {
		return RequestResult{Index: index, Endpoint: endpoint, LatencyMS: time.Since(start).Milliseconds(), Error: fmt.Sprintf("send request: %v", err)}
	}
	defer response.Body.Close()

	result := RequestResult{
		Index:      index,
		Endpoint:   endpoint,
		StatusCode: response.StatusCode,
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		snippet, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
		result.LatencyMS = time.Since(start).Milliseconds()
		result.Error = fmt.Sprintf("backend returned %s: %s", response.Status, bytes.TrimSpace(snippet))
		return result
	}

	if request.Streaming {
		metrics, err := consumeSSE(response.Body, start, time.Now)
		result.LatencyMS = time.Since(start).Milliseconds()
		result.TTFTMS = metrics.TTFTMS
		result.InterTokenLatencyMS = metrics.InterTokenLatencyMS
		result.OutputTokens = metrics.OutputTokens
		result.Trace = metrics.Trace
		if err != nil {
			result.Error = fmt.Sprintf("decode stream: %v", err)
			return result
		}
		return result
	}

	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		result.LatencyMS = time.Since(start).Milliseconds()
		result.Error = fmt.Sprintf("decode response: %v", err)
		return result
	}
	result.LatencyMS = time.Since(start).Milliseconds()
	if decoded.Usage != nil {
		result.OutputTokens = decoded.Usage.CompletionTokens
	}
	result.Route = decoded.Route
	result.Trace = decoded.Trace
	return result
}

func consumeSSE(reader io.Reader, startedAt time.Time, now func() time.Time) (streamMetrics, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	metrics := streamMetrics{InterTokenLatencyMS: make([]int64, 0)}
	var dataLines []string
	var previousTokenAt time.Time
	sawDone := false
	sawUsage := false

	recordToken := func() {
		tokenAt := now()
		if metrics.OutputTokens == 0 {
			metrics.TTFTMS = tokenAt.Sub(startedAt).Milliseconds()
		} else {
			metrics.InterTokenLatencyMS = append(
				metrics.InterTokenLatencyMS,
				tokenAt.Sub(previousTokenAt).Milliseconds(),
			)
		}
		metrics.OutputTokens++
		previousTokenAt = tokenAt
	}

	consumeEvent := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		if payload == "[DONE]" {
			sawDone = true
			return nil
		}
		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return fmt.Errorf("decode SSE data: %w", err)
		}
		if chunk.Error != nil {
			return fmt.Errorf("%s: %s", chunk.Error.Code, chunk.Error.Message)
		}
		if chunk.Usage != nil {
			if chunk.Usage.CompletionTokens < 0 {
				return fmt.Errorf("negative completion token count")
			}
			if chunk.Usage.CompletionTokens != metrics.OutputTokens {
				return fmt.Errorf(
					"authoritative completion tokens=%d, observed token events=%d",
					chunk.Usage.CompletionTokens,
					metrics.OutputTokens,
				)
			}
			sawUsage = true
		}
		if chunk.Trace != nil && len(chunk.Trace.StageTimings) > 0 {
			metrics.Trace = &chat.RuntimeTrace{
				StageCalls:       chunk.Trace.StageCalls,
				RemoteStageCalls: chunk.Trace.RemoteStageCalls,
				BytesIn:          chunk.Trace.BytesIn,
				BytesOut:         chunk.Trace.BytesOut,
				StageTimings:     append([]chat.StageTiming(nil), chunk.Trace.StageTimings...),
			}
		}
		if chunk.Trace != nil && chunk.Trace.TokenIndex != nil {
			if *chunk.Trace.TokenIndex != metrics.OutputTokens {
				return fmt.Errorf(
					"stream token index=%d, want %d",
					*chunk.Trace.TokenIndex,
					metrics.OutputTokens,
				)
			}
			recordToken()
			return nil
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == nil || *choice.Delta.Content == "" {
				continue
			}
			recordToken()
		}
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := consumeEvent(); err != nil {
				return metrics, err
			}
			if sawDone {
				return metrics, nil
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return metrics, err
	}
	if err := consumeEvent(); err != nil {
		return metrics, err
	}
	if !sawDone {
		return metrics, fmt.Errorf("stream ended before data: [DONE]")
	}
	if !sawUsage {
		return metrics, fmt.Errorf("stream ended without authoritative token usage")
	}
	return metrics, nil
}

func summarize(endpoint string, streaming bool, count int, concurrency int, startedAt time.Time, finishedAt time.Time, results []RequestResult) Summary {
	successes := 0
	failures := 0
	outputTokens := 0
	latencies := make([]int64, 0, len(results))
	ttfts := make([]int64, 0, len(results))
	interTokenLatencies := make([]int64, 0)
	for _, result := range results {
		if result.Error != "" {
			failures++
			continue
		}
		successes++
		outputTokens += result.OutputTokens
		latencies = append(latencies, result.LatencyMS)
		if streaming && result.OutputTokens > 0 {
			ttfts = append(ttfts, result.TTFTMS)
			interTokenLatencies = append(interTokenLatencies, result.InterTokenLatencyMS...)
		}
	}

	totalDuration := finishedAt.Sub(startedAt)
	requestThroughput := 0.0
	outputThroughput := 0.0
	if outputTokens > 0 && totalDuration > 0 {
		outputThroughput = float64(outputTokens) / totalDuration.Seconds()
	}
	if successes > 0 && totalDuration > 0 {
		requestThroughput = float64(successes) / totalDuration.Seconds()
	}
	meanOutputTokens := 0.0
	if successes > 0 {
		meanOutputTokens = float64(outputTokens) / float64(successes)
	}

	return Summary{
		Endpoint:          endpoint,
		Streaming:         streaming,
		RequestCount:      count,
		Concurrency:       concurrency,
		SuccessCount:      successes,
		FailureCount:      failures,
		StartedAt:         startedAt,
		FinishedAt:        finishedAt,
		TotalDurationMS:   totalDuration.Milliseconds(),
		Latency:           summarizeLatencies(latencies),
		TTFT:              summarizeLatencies(ttfts),
		InterTokenLatency: summarizeLatencies(interTokenLatencies),
		OutputTokens:      outputTokens,
		MeanOutputTokens:  meanOutputTokens,
		RequestThroughput: requestThroughput,
		OutputThroughput:  outputThroughput,
		TokensPerSecond:   outputThroughput,
		Runtime:           summarizeRuntime(results),
		Results:           results,
	}
}

func summarizeRuntime(results []RequestResult) RuntimeSummary {
	summary := RuntimeSummary{}
	timings := make([]chat.StageTiming, 0)
	indexByKey := make(map[string]int)
	for _, result := range results {
		if result.Error != "" || result.Trace == nil {
			continue
		}
		summary.StageCalls += result.Trace.StageCalls
		summary.RemoteStageCalls += result.Trace.RemoteStageCalls
		summary.BytesIn += result.Trace.BytesIn
		summary.BytesOut += result.Trace.BytesOut
		for _, timing := range result.Trace.StageTimings {
			key := fmt.Sprintf("%s:%d:%s", timing.Phase, timing.StageIndex, timing.NodeName)
			index, exists := indexByKey[key]
			if !exists {
				index = len(timings)
				indexByKey[key] = index
				timings = append(timings, chat.StageTiming{
					Phase:      timing.Phase,
					StageIndex: timing.StageIndex,
					NodeName:   timing.NodeName,
					Remote:     timing.Remote,
				})
			}
			aggregate := &timings[index]
			aggregate.Calls += timing.Calls
			aggregate.ExecutionUS += timing.ExecutionUS
			aggregate.ActivationDecodeUS += timing.ActivationDecodeUS
			aggregate.ActivationEncodeUS += timing.ActivationEncodeUS
			aggregate.StageTotalUS += timing.StageTotalUS
			aggregate.RemoteCallUS += timing.RemoteCallUS
			aggregate.RemoteOverheadUS += timing.RemoteOverheadUS
			aggregate.BytesIn += timing.BytesIn
			aggregate.BytesOut += timing.BytesOut
		}
	}
	for _, timing := range timings {
		summary.StageTimings = append(summary.StageTimings, StageTimingSummary{
			Phase:            timing.Phase,
			StageIndex:       timing.StageIndex,
			NodeName:         timing.NodeName,
			Remote:           timing.Remote,
			Calls:            timing.Calls,
			Execution:        microsecondTotal(timing.ExecutionUS, timing.Calls),
			ActivationDecode: microsecondTotal(timing.ActivationDecodeUS, timing.Calls),
			ActivationEncode: microsecondTotal(timing.ActivationEncodeUS, timing.Calls),
			StageTotal:       microsecondTotal(timing.StageTotalUS, timing.Calls),
			RemoteCall:       microsecondTotal(timing.RemoteCallUS, timing.Calls),
			RemoteOverhead:   microsecondTotal(timing.RemoteOverheadUS, timing.Calls),
			BytesIn:          timing.BytesIn,
			BytesOut:         timing.BytesOut,
		})
	}
	return summary
}

func microsecondTotal(total int64, calls int) MicrosecondTotal {
	average := 0.0
	if calls > 0 {
		average = float64(total) / float64(calls)
	}
	return MicrosecondTotal{TotalUS: total, AvgUS: average}
}

func summarizeLatencies(values []int64) LatencySummary {
	if len(values) == 0 {
		return LatencySummary{}
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i int, j int) bool { return sorted[i] < sorted[j] })
	total := int64(0)
	for _, value := range sorted {
		total += value
	}
	return LatencySummary{
		MinMS: sorted[0],
		MaxMS: sorted[len(sorted)-1],
		AvgMS: float64(total) / float64(len(sorted)),
		P50MS: nearestRank(sorted, 0.50),
		P90MS: nearestRank(sorted, 0.90),
		P95MS: nearestRank(sorted, 0.95),
		P99MS: nearestRank(sorted, 0.99),
	}
}

func nearestRank(sorted []int64, percentile float64) int64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func writeOutput(path string, content []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), outputDirPerm); err != nil {
		return err
	}
	return os.WriteFile(path, content, outputFilePerm)
}

func writeJSONOutput(value any, path string) error {
	content, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	content = append(content, '\n')
	if path != "" {
		return writeOutput(path, content)
	}
	_, err = os.Stdout.Write(content)
	return err
}
