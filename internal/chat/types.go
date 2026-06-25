package chat

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
	Route   *RouteMetadata `json:"jetsonmesh_route,omitempty"`
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
	Mode        string `json:"mode"`
	NodeID      string `json:"node_id"`
	BackendID   string `json:"backend_id"`
	BackendKind string `json:"backend_kind"`
	LatencyMS   int64  `json:"latency_ms"`
}
