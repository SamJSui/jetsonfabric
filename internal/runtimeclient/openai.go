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

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/chat"
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
	chatCompletionsURL string
	http               *http.Client
}

func NewOpenAIClient(baseURL string, timeout time.Duration) (*OpenAIClient, error) {
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	chatCompletionsURL, err := api.JoinBasePath(baseURL, api.PathChatCompletions)
	if err != nil {
		return nil, err
	}
	return &OpenAIClient{
		chatCompletionsURL: chatCompletionsURL,
		http:               &http.Client{Timeout: timeout},
	}, nil
}

func (c *OpenAIClient) Complete(ctx context.Context, req chat.CompletionRequest) (chat.CompletionResponse, Stats, error) {
	start := time.Now()
	body, err := json.Marshal(req)
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("encode chat request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatCompletionsURL, bytes.NewReader(body))
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("create backend request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("call backend %s: %w", c.chatCompletionsURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("backend %s returned %s: %s", c.chatCompletionsURL, resp.Status, strings.TrimSpace(string(snippet)))
	}

	var decoded chat.CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("decode backend response: %w", err)
	}
	if decoded.Model == "" {
		decoded.Model = req.Model
	}
	if len(decoded.Choices) == 0 {
		return chat.CompletionResponse{}, Stats{}, fmt.Errorf("backend %s returned no choices", c.chatCompletionsURL)
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
