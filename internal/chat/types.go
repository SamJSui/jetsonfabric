package chat

import "github.com/SamJSui/jetsonfabric/internal/cluster"

type CompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionResponse struct {
	ID      string         `json:"id,omitempty"`
	Object  string         `json:"object,omitempty"`
	Created int64          `json:"created,omitempty"`
	Model   string         `json:"model"`
	Choices []Choice       `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
	Route   *RouteMetadata `json:"jetsonfabric_route,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type RouteMetadata struct {
	Mode        cluster.RouteMode   `json:"mode"`
	NodeName    string              `json:"node_name"`
	BackendID   string              `json:"backend_id"`
	BackendKind cluster.RuntimeKind `json:"backend_kind"`
	LatencyMS   int64               `json:"latency_ms"`
	Stages      []RouteStage        `json:"stages,omitempty"`
	BytesIn     int                 `json:"bytes_in,omitempty"`
	BytesOut    int                 `json:"bytes_out,omitempty"`
}

type RouteStage struct {
	Index       int                 `json:"index"`
	NodeName    string              `json:"node_name"`
	BackendID   string              `json:"backend_id,omitempty"`
	BackendKind cluster.RuntimeKind `json:"backend_kind,omitempty"`
	Role        string              `json:"role"`
	LayerStart  int                 `json:"layer_start"`
	LayerEnd    int                 `json:"layer_end"`
	Transport   string              `json:"transport"`
	LatencyMS   int64               `json:"latency_ms"`
	BytesIn     int                 `json:"bytes_in"`
	BytesOut    int                 `json:"bytes_out"`
}
