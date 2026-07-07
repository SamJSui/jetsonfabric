package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
)

const (
	defaultEndpoint    = "http://127.0.0.1:52415" + api.PathChatCompletions
	defaultRequestPath = "examples/poc-local-smoke/chat-request.json"
	defaultBenchCount  = 1
	defaultConcurrency = 1
	outputFilePerm     = 0o644
	outputDirPerm      = 0o755
	maxErrorBodyBytes  = 4096
)

type Summary struct {
	Endpoint        string          `json:"endpoint"`
	RequestCount    int             `json:"request_count"`
	Concurrency     int             `json:"concurrency"`
	SuccessCount    int             `json:"success_count"`
	FailureCount    int             `json:"failure_count"`
	StartedAt       time.Time       `json:"started_at"`
	FinishedAt      time.Time       `json:"finished_at"`
	TotalDurationMS int64           `json:"total_duration_ms"`
	Latency         LatencySummary  `json:"latency"`
	OutputTokens    int             `json:"output_tokens"`
	TokensPerSecond float64         `json:"tokens_per_second"`
	Results         []RequestResult `json:"results"`
}

type LatencySummary struct {
	MinMS int64   `json:"min_ms"`
	MaxMS int64   `json:"max_ms"`
	AvgMS float64 `json:"avg_ms"`
}

type RequestResult struct {
	Index        int                 `json:"index"`
	StatusCode   int                 `json:"status_code,omitempty"`
	LatencyMS    int64               `json:"latency_ms"`
	OutputTokens int                 `json:"output_tokens,omitempty"`
	Route        *chat.RouteMetadata `json:"jetsonfabric_route,omitempty"`
	Error        string              `json:"error,omitempty"`
}

func main() {
	endpoint := flag.String("url", defaultEndpoint, "chat completions endpoint URL")
	requestPath := flag.String("request", defaultRequestPath, "chat request JSON path")
	count := flag.Int("count", defaultBenchCount, "number of requests to send")
	concurrency := flag.Int("concurrency", defaultConcurrency, "number of concurrent workers")
	timeout := flag.Duration("timeout", 2*time.Minute, "per-request timeout")
	outputPath := flag.String("output", "", "optional JSON output path")
	flag.Parse()

	request, err := loadCompletionRequest(*requestPath)
	if err != nil {
		log.Fatalf("load request: %v", err)
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

func loadCompletionRequest(path string) (chat.CompletionRequest, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return chat.CompletionRequest{}, err
	}
	var request chat.CompletionRequest
	if err := json.Unmarshal(content, &request); err != nil {
		return chat.CompletionRequest{}, err
	}
	if request.Model == "" {
		return chat.CompletionRequest{}, fmt.Errorf("request model is required")
	}
	if len(request.Messages) == 0 {
		return chat.CompletionRequest{}, fmt.Errorf("request messages are required")
	}
	return request, nil
}

func runBenchmark(
	ctx context.Context,
	client *http.Client,
	endpoint string,
	request chat.CompletionRequest,
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

	return summarize(endpoint, count, concurrency, startedAt, finishedAt, results), nil
}

func sendRequest(ctx context.Context, client *http.Client, endpoint string, request chat.CompletionRequest, index int) RequestResult {
	body, err := json.Marshal(request)
	if err != nil {
		return RequestResult{Index: index, Error: fmt.Sprintf("encode request: %v", err)}
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return RequestResult{Index: index, Error: fmt.Sprintf("create request: %v", err)}
	}
	httpRequest.Header.Set("Content-Type", "application/json")

	start := time.Now()
	response, err := client.Do(httpRequest)
	elapsed := time.Since(start)
	if err != nil {
		return RequestResult{Index: index, LatencyMS: elapsed.Milliseconds(), Error: fmt.Sprintf("send request: %v", err)}
	}
	defer response.Body.Close()

	result := RequestResult{
		Index:      index,
		StatusCode: response.StatusCode,
		LatencyMS:  elapsed.Milliseconds(),
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		snippet, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBodyBytes))
		result.Error = fmt.Sprintf("backend returned %s: %s", response.Status, bytes.TrimSpace(snippet))
		return result
	}

	var decoded chat.CompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		result.Error = fmt.Sprintf("decode response: %v", err)
		return result
	}
	if decoded.Usage != nil {
		result.OutputTokens = decoded.Usage.CompletionTokens
	}
	result.Route = decoded.Route
	return result
}

func summarize(endpoint string, count int, concurrency int, startedAt time.Time, finishedAt time.Time, results []RequestResult) Summary {
	successes := 0
	failures := 0
	outputTokens := 0
	totalLatency := int64(0)
	minLatency := int64(0)
	maxLatency := int64(0)
	for _, result := range results {
		if result.Error != "" {
			failures++
			continue
		}
		successes++
		outputTokens += result.OutputTokens
		totalLatency += result.LatencyMS
		if minLatency == 0 || result.LatencyMS < minLatency {
			minLatency = result.LatencyMS
		}
		if result.LatencyMS > maxLatency {
			maxLatency = result.LatencyMS
		}
	}

	avgLatency := 0.0
	if successes > 0 {
		avgLatency = float64(totalLatency) / float64(successes)
	}
	totalDuration := finishedAt.Sub(startedAt)
	tokensPerSecond := 0.0
	if outputTokens > 0 && totalDuration > 0 {
		tokensPerSecond = float64(outputTokens) / totalDuration.Seconds()
	}

	return Summary{
		Endpoint:        endpoint,
		RequestCount:    count,
		Concurrency:     concurrency,
		SuccessCount:    successes,
		FailureCount:    failures,
		StartedAt:       startedAt,
		FinishedAt:      finishedAt,
		TotalDurationMS: totalDuration.Milliseconds(),
		Latency: LatencySummary{
			MinMS: minLatency,
			MaxMS: maxLatency,
			AvgMS: avgLatency,
		},
		OutputTokens:    outputTokens,
		TokensPerSecond: tokensPerSecond,
		Results:         results,
	}
}

func writeOutput(path string, content []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), outputDirPerm); err != nil {
		return err
	}
	return os.WriteFile(path, content, outputFilePerm)
}
