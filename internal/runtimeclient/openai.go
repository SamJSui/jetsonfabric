package runtimeclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/chat"
)

type Stats struct {
	Latency      time.Duration
	OutputTokens int
	TokensPerSec float64
}

type ChatBackend interface {
	Complete(ctx context.Context, req chat.CompletionRequest) (chat.CompletionResponse, Stats, error)
}

type OpenAIClient struct {
	baseURL string
	http    *http.Client
}

func NewOpenAIClient(baseURL string, timeout time.Duration) *OpenAIClient {
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &OpenAIClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: timeout},
	}
}

func (c *OpenAIClient) Complete(ctx context.Context, req chat.CompletionRequest) (chat.CompletionResponse, Stats, error) {
	start := time.Now()
	body, err := json.Marshal(req)
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("encode chat request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("create backend request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("call backend %s: %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("backend %s returned %s: %s", c.baseURL, resp.Status, strings.TrimSpace(string(snippet)))
	}

	var decoded chat.CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("decode backend response: %w", err)
	}
	if decoded.Model == "" {
		decoded.Model = req.Model
	}
	if len(decoded.Choices) == 0 {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("backend %s returned no choices", c.baseURL)
	}

	elapsed := time.Since(start)
	outputTokens := 0
	if decoded.Usage != nil {
		outputTokens = decoded.Usage.CompletionTokens
	}
	tokensPerSec := 0.0
	if outputTokens > 0 && elapsed > 0 {
		tokensPerSec = float64(outputTokens) / elapsed.Seconds()
	}
	return decoded, Stats{
		Latency:      elapsed,
		OutputTokens: outputTokens,
		TokensPerSec: tokensPerSec,
	}, nil
}
