package runtimebridge

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
	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

const defaultChatModel = "qwen2.5-coder-1.5b-q4"

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
	cfg = normalizeChatProxyConfig(cfg)
	return &ChatProxy{
		runtimeURL: parsed,
		client:     &http.Client{Timeout: 5 * time.Minute},
		nodeName:   cfg.NodeName,
		model:      cfg.Model,
		layerStart: cfg.LayerStart,
		layerEnd:   cfg.LayerEnd,
	}, nil
}

func normalizeChatProxyConfig(cfg ChatProxyConfig) ChatProxyConfig {
	if cfg.NodeName == "" {
		cfg.NodeName = "node"
	}
	if cfg.Model == "" {
		cfg.Model = defaultChatModel
	}
	if cfg.LayerEnd <= cfg.LayerStart {
		cfg.LayerStart = 0
		cfg.LayerEnd = 28
	}
	return cfg
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

type preparedChatRequest struct {
	RequestID string
	Model     string
	Stage     stagewire.StageRequest
}

type chatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (p *ChatProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorPayload("method_not_allowed", "chat completions requires POST"))
		return
	}
	chatReq, ok := decodeChatCompletionRequest(w, req)
	if !ok {
		return
	}
	prepared, ok := p.prepareChatRequest(w, chatReq)
	if !ok {
		return
	}
	stageResp, status, err := p.callRuntime(req, prepared.Stage)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_chat_failed", err.Error()))
		return
	}
	if !runtimeStatusOK(w, status, stageResp) {
		return
	}
	writeJSON(w, http.StatusOK, makeChatCompletionResponse(prepared, stageResp))
}

func decodeChatCompletionRequest(w http.ResponseWriter, req *http.Request) (chatCompletionRequest, bool) {
	var chatReq chatCompletionRequest
	if err := json.NewDecoder(req.Body).Decode(&chatReq); err != nil {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_request", err.Error()))
		return chatCompletionRequest{}, false
	}
	return chatReq, true
}

func (p *ChatProxy) prepareChatRequest(w http.ResponseWriter, chatReq chatCompletionRequest) (preparedChatRequest, bool) {
	if chatReq.Stream {
		writeJSON(w, http.StatusBadRequest, errorPayload("stream_not_supported", "streaming is not implemented yet"))
		return preparedChatRequest{}, false
	}
	if !hasChatMessages(chatReq.Messages) {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_request", "messages are required"))
		return preparedChatRequest{}, false
	}
	requestID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	model := p.requestModel(chatReq.Model)
	prompt := renderQwenPrompt(chatReq.Messages)
	stageReq := p.buildStageRequest(requestID, model, prompt, chatReq.MaxTokens)
	return preparedChatRequest{RequestID: requestID, Model: model, Stage: stageReq}, true
}

func hasChatMessages(messages []chatMessage) bool {
	for _, message := range messages {
		if strings.TrimSpace(message.Content) != "" {
			return true
		}
	}
	return false
}

func (p *ChatProxy) requestModel(model string) string {
	model = strings.TrimSpace(model)
	if model == "" {
		return p.model
	}
	return model
}

func (p *ChatProxy) buildStageRequest(requestID, model, prompt string, maxTokens int) stagewire.StageRequest {
	return stagewire.StageRequest{
		Metadata: stagewire.Metadata{
			SessionID: requestID, RequestID: requestID, ModelID: model,
			Phase: inference.PhasePrefill, DecodeStep: 0,
			StageIndex: 0, StageCount: 1, NodeName: p.nodeName,
			LayerStart: p.layerStart, LayerEnd: p.layerEnd,
			PayloadKind: stagewire.PayloadKindText, Encoding: "utf-8",
			MaxTokens: normalizeMaxTokens(maxTokens),
		},
		Payload: []byte(prompt),
	}
}

func normalizeMaxTokens(maxTokens int) int {
	if maxTokens <= 0 {
		return 128
	}
	if maxTokens > 1024 {
		return 1024
	}
	return maxTokens
}

func runtimeStatusOK(w http.ResponseWriter, status int, stageResp stagewire.StageResponse) bool {
	if status >= 200 && status < 300 {
		return true
	}
	if stageResp.Error == "" {
		stageResp.Error = "runtime_error"
	}
	writeJSON(w, status, errorPayload(stageResp.Error, stageResp.Message))
	return false
}

func makeChatCompletionResponse(prepared preparedChatRequest, stageResp stagewire.StageResponse) chatCompletionResponse {
	return chatCompletionResponse{
		ID: prepared.RequestID, Object: "chat.completion", Created: time.Now().Unix(), Model: prepared.Model,
		Choices: []chatChoice{{
			Index:        0,
			Message:      chatMessage{Role: "assistant", Content: strings.TrimSpace(string(stageResp.Payload))},
			FinishReason: "stop",
		}},
		Usage: usageFromStageResponse(stageResp),
	}
}

func usageFromStageResponse(stageResp stagewire.StageResponse) chatUsage {
	return chatUsage{
		PromptTokens: stageResp.PromptTokens, CompletionTokens: stageResp.CompletionTokens,
		TotalTokens: stageResp.PromptTokens + stageResp.CompletionTokens,
	}
}

func (p *ChatProxy) callRuntime(req *http.Request, stageReq stagewire.StageRequest) (stagewire.StageResponse, int, error) {
	body, err := stagewire.Marshal(stageReq)
	if err != nil {
		return stagewire.StageResponse{}, 0, err
	}
	outbound, err := p.newRuntimeStageRequest(req, body)
	if err != nil {
		return stagewire.StageResponse{}, 0, err
	}
	return p.doRuntimeStageRequest(outbound)
}

func (p *ChatProxy) newRuntimeStageRequest(req *http.Request, body []byte) (*http.Request, error) {
	outbound, err := http.NewRequestWithContext(req.Context(), http.MethodPost, p.runtimeStageURL(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	outbound.Header.Set("Content-Type", stagewire.ContentType)
	outbound.Header.Set("Accept", stagewire.ContentType)
	outbound.ContentLength = int64(len(body))
	return outbound, nil
}

func (p *ChatProxy) runtimeStageURL() string {
	target := *p.runtimeURL
	target.Path = joinPath(target.Path, api.PathLayerSplitStage)
	return target.String()
}

func (p *ChatProxy) doRuntimeStageRequest(outbound *http.Request) (stagewire.StageResponse, int, error) {
	resp, err := p.client.Do(outbound)
	if err != nil {
		return stagewire.StageResponse{}, 0, err
	}
	defer resp.Body.Close()
	stageResp, err := decodeRuntimeStageResponse(resp)
	return stageResp, resp.StatusCode, err
}

func decodeRuntimeStageResponse(resp *http.Response) (stagewire.StageResponse, error) {
	if strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), strings.ToLower(stagewire.ContentType)) {
		return stagewire.Decode(resp.Body)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return stagewire.StageResponse{}, err
	}
	var failure struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &failure); err != nil {
		return stagewire.StageResponse{}, fmt.Errorf("decode runtime response: %w: %s", err, string(body))
	}
	return stagewire.StageResponse{Metadata: stagewire.Metadata{Error: failure.Error, Message: failure.Message}}, nil
}

func renderQwenPrompt(messages []chatMessage) string {
	var b strings.Builder
	for _, message := range messages {
		appendQwenMessage(&b, message)
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func appendQwenMessage(b *strings.Builder, message chatMessage) {
	role := strings.TrimSpace(message.Role)
	content := strings.TrimSpace(message.Content)
	if content == "" {
		return
	}
	if role == "" {
		role = "user"
	}
	b.WriteString("<|im_start|>")
	b.WriteString(role)
	b.WriteString("\n")
	b.WriteString(content)
	b.WriteString("<|im_end|>\n")
}
