package runtimegateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

type ChatProxyConfig struct {
	RuntimeURL string
	NodeName   string
	Model      string
	LayerStart int
	LayerEnd   int
}

type ChatProxy struct {
	runtimeURL *url.URL
	client     *http.Client

	nodeName   string
	model      string
	layerStart int
	layerEnd   int
}

func NewChatProxy(cfg ChatProxyConfig) (*ChatProxy, error) {
	parsed, err := parseRuntimeURL(cfg.RuntimeURL)
	if err != nil {
		return nil, err
	}

	if cfg.NodeName == "" {
		cfg.NodeName = "node"
	}
	if cfg.Model == "" {
		cfg.Model = "qwen2.5-coder-1.5b-q4"
	}
	if cfg.LayerEnd <= cfg.LayerStart {
		cfg.LayerStart = 0
		cfg.LayerEnd = 28
	}

	return &ChatProxy{
		runtimeURL: parsed,
		client:     &http.Client{Timeout: 5 * time.Minute},
		nodeName:   cfg.NodeName,
		model:      cfg.Model,
		layerStart: cfg.LayerStart,
		layerEnd:   cfg.LayerEnd,
	}, nil
}

type chatCompletionRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens,omitempty"`
	Stream    bool          `json:"stream,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type stageRequest struct {
	SessionID  string `json:"session_id"`
	RequestID  string `json:"request_id"`
	ModelID    string `json:"model_id"`
	StageIndex int    `json:"stage_index"`
	NodeName   string `json:"node_name"`
	Role       string `json:"role"`

	LayerStart int `json:"layer_start"`
	LayerEnd   int `json:"layer_end"`
	DecodeStep int `json:"decode_step"`

	Shape     []int  `json:"shape"`
	DType     string `json:"dtype"`
	Payload   string `json:"payload"`
	BytesIn   int    `json:"bytes_in"`
	Transport string `json:"transport"`

	MaxTokens int `json:"max_tokens,omitempty"`
}

type stageResponse struct {
	Payload  string `json:"payload"`
	BytesIn  int    `json:"bytes_in"`
	BytesOut int    `json:"bytes_out"`
	Error    string `json:"error,omitempty"`
	Message  string `json:"message,omitempty"`
}

type chatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

func (p *ChatProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorPayload("method_not_allowed", "chat completions requires POST"))
		return
	}

	var chatReq chatCompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&chatReq); err != nil {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_request", err.Error()))
		return
	}

	if chatReq.Stream {
		writeJSON(w, http.StatusBadRequest, errorPayload("stream_not_supported", "streaming is not implemented yet"))
		return
	}

	model := strings.TrimSpace(chatReq.Model)
	if model == "" {
		model = p.model
	}

	maxTokens := chatReq.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}
	if maxTokens > 1024 {
		maxTokens = 1024
	}

	prompt := renderQwenPrompt(chatReq.Messages)
	if strings.TrimSpace(prompt) == "" {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_request", "messages are required"))
		return
	}

	requestID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	stageReq := stageRequest{
		SessionID:  requestID,
		RequestID:  requestID,
		ModelID:    model,
		StageIndex: 0,
		NodeName:   p.nodeName,
		Role:       "single",

		LayerStart: p.layerStart,
		LayerEnd:   p.layerEnd,
		DecodeStep: 0,

		Shape:     []int{1, p.layerEnd - p.layerStart, 1},
		DType:     "text",
		Payload:   prompt,
		BytesIn:   len([]byte(prompt)),
		Transport: "http",

		MaxTokens: maxTokens,
	}

	stageResp, status, err := p.callRuntime(req, stageReq)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_chat_failed", err.Error()))
		return
	}
	if status < 200 || status >= 300 {
		writeJSON(w, status, errorPayload(stageResp.Error, stageResp.Message))
		return
	}

	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []chatChoice{
			{
				Index: 0,
				Message: chatMessage{
					Role:    "assistant",
					Content: strings.TrimSpace(stageResp.Payload),
				},
				FinishReason: "stop",
			},
		},
	})
}

func (p *ChatProxy) callRuntime(req *http.Request, stageReq stageRequest) (stageResponse, int, error) {
	target := *p.runtimeURL
	target.Path = joinPath(target.Path, api.PathLayerSplitStage)

	body, err := json.Marshal(stageReq)
	if err != nil {
		return stageResponse{}, 0, err
	}

	outbound, err := http.NewRequestWithContext(
		req.Context(),
		http.MethodPost,
		target.String(),
		bytes.NewReader(body),
	)
	if err != nil {
		return stageResponse{}, 0, err
	}

	outbound.Header.Set("Content-Type", "application/json")
	outbound.ContentLength = int64(len(body))

	resp, err := p.client.Do(outbound)
	if err != nil {
		return stageResponse{}, 0, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return stageResponse{}, resp.StatusCode, err
	}

	var stageResp stageResponse
	if err := json.Unmarshal(respBody, &stageResp); err != nil {
		return stageResponse{}, resp.StatusCode, fmt.Errorf("decode runtime response: %w: %s", err, string(respBody))
	}

	return stageResp, resp.StatusCode, nil
}

func renderQwenPrompt(messages []chatMessage) string {
	var b strings.Builder

	for _, message := range messages {
		role := strings.TrimSpace(message.Role)
		if role == "" {
			role = "user"
		}

		content := strings.TrimSpace(message.Content)
		if content == "" {
			continue
		}

		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}

	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}
