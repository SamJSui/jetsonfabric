package main

import (
	"bufio"
	"bytes"
	"context"
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
	Endpoint          string          `json:"endpoint"`
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
	TokensPerSecond   float64         `json:"tokens_per_second"`
	Results           []RequestResult `json:"results"`
}

type LatencySummary struct {
	MinMS int64   `json:"min_ms"`
	MaxMS int64   `json:"max_ms"`
	AvgMS float64 `json:"avg_ms"`
	P50MS int64   `json:"p50_ms"`
	P95MS int64   `json:"p95_ms"`
}

type RequestResult struct {
	Index               int                 `json:"index"`
	StatusCode          int                 `json:"status_code,omitempty"`
	LatencyMS           int64               `json:"latency_ms"`
	TTFTMS              int64               `json:"ttft_ms,omitempty"`
	InterTokenLatencyMS []int64             `json:"inter_token_latency_ms,omitempty"`
	OutputTokens        int                 `json:"output_tokens,omitempty"`
	Route               *chat.RouteMetadata `json:"jetsonfabric_route,omitempty"`
	Error               string              `json:"error,omitempty"`
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
	Error *struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

type streamMetrics struct {
	TTFTMS              int64
	InterTokenLatencyMS []int64
	OutputTokens        int
}

func main() {
	endpoint := flag.String("url", defaultEndpoint, "chat completions endpoint URL")
	requestPath := flag.String("request", defaultRequestPath, "chat request JSON path")
	count := flag.Int("count", defaultBenchCount, "number of requests to send")
	concurrency := flag.Int("concurrency", defaultConcurrency, "number of concurrent workers")
	timeout := flag.Duration("timeout", 2*time.Minute, "per-request timeout")
	stream := flag.Bool("stream", false, "force stream=true and report TTFT/inter-token latency")
	outputPath := flag.String("output", "", "optional JSON output path")
	flag.Parse()

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
	summary, err := runBenchmark(context.Background(), http.DefaultClient, *endpoint, request, *count, *concurrency, *timeout)
	if err != nil {
		log.Fatalf("run benchmark: %v", err)
	}
	content, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		log.Fatalf("encode summary: %v", err)
	}
	content = append(content, '\n')

	if *outputPath != "" {
		if err := writeOutput(*outputPath, content); err != nil {
			log.Fatalf("write output: %v", err)
		}
		return
	}
	if _, err := os.Stdout.Write(content); err != nil {
		log.Fatalf("write stdout: %v", err)
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

func runBenchmark(
	ctx context.Context,
	client *http.Client,
	endpoint string,
	request benchmarkRequest,
	count int,
	concurrency int,
	timeout time.Duration,
) (Summary, error) {
	if count <= 0 {
		return Summary{}, fmt.Errorf("count must be greater than zero")
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

	startedAt := time.Now().UTC()
	results := make([]RequestResult, count)
	jobs := make(chan int)
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				requestCtx, cancel := context.WithTimeout(ctx, timeout)
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
	finishedAt := time.Now().UTC()

	return summarize(endpoint, request.Streaming, count, concurrency, startedAt, finishedAt, results), nil
}

func sendRequest(ctx context.Context, client *http.Client, endpoint string, request benchmarkRequest, index int) RequestResult {
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(request.Body))
	if err != nil {
		return RequestResult{Index: index, Error: fmt.Sprintf("create request: %v", err)}
	}
	httpRequest.Header.Set("Content-Type", "application/json")

	start := time.Now()
	response, err := client.Do(httpRequest)
	if err != nil {
		return RequestResult{Index: index, LatencyMS: time.Since(start).Milliseconds(), Error: fmt.Sprintf("send request: %v", err)}
	}
	defer response.Body.Close()

	result := RequestResult{
		Index:      index,
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
	return result
}

func consumeSSE(reader io.Reader, startedAt time.Time, now func() time.Time) (streamMetrics, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	metrics := streamMetrics{InterTokenLatencyMS: make([]int64, 0)}
	var dataLines []string
	var previousTokenAt time.Time
	sawDone := false

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
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == nil || *choice.Delta.Content == "" {
				continue
			}
			tokenAt := now()
			if metrics.OutputTokens == 0 {
				metrics.TTFTMS = tokenAt.Sub(startedAt).Milliseconds()
			} else {
				metrics.InterTokenLatencyMS = append(metrics.InterTokenLatencyMS, tokenAt.Sub(previousTokenAt).Milliseconds())
			}
			metrics.OutputTokens++
			previousTokenAt = tokenAt
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
		if streaming {
			ttfts = append(ttfts, result.TTFTMS)
			interTokenLatencies = append(interTokenLatencies, result.InterTokenLatencyMS...)
		}
	}

	totalDuration := finishedAt.Sub(startedAt)
	tokensPerSecond := 0.0
	if outputTokens > 0 && totalDuration > 0 {
		tokensPerSecond = float64(outputTokens) / totalDuration.Seconds()
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
		TokensPerSecond:   tokensPerSecond,
		Results:           results,
	}
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
		P95MS: nearestRank(sorted, 0.95),
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
