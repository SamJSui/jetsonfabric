package coordinator

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

type chatCompletionRequest struct {
	Model               string                  `json:"model"`
	Messages            []chatCompletionMessage `json:"messages"`
	MaxTokens           int                     `json:"max_tokens,omitempty"`
	MaxCompletionTokens int                     `json:"max_completion_tokens,omitempty"`
	Stream              bool                    `json:"stream,omitempty"`
	JetsonFabric        *chatCompletionRouting  `json:"jetsonfabric,omitempty"`
}

type chatCompletionRouting struct {
	StageCount           int  `json:"stage_count,omitempty"`
	AllowColocatedStages bool `json:"allow_colocated_stages,omitempty"`
}

type chatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatCompletionUsage    `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      chatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIErrorEnvelope struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var request chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", nil, err.Error())
		return
	}
	if request.JetsonFabric != nil && request.JetsonFabric.StageCount < 0 {
		param := "jetsonfabric.stage_count"
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_stage_count", &param, "stage_count must be greater than zero")
		return
	}
	modelID := strings.TrimSpace(request.Model)
	if modelID == "" {
		param := "model"
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "model_required", &param, "model is required")
		return
	}
	prompt := renderChatPrompt(request.Messages)
	if prompt == "" {
		param := "messages"
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "messages_required", &param, "at least one non-empty message is required")
		return
	}
	policy := s.routePreviewPolicy(r)
	if request.JetsonFabric != nil {
		if request.JetsonFabric.StageCount > 0 {
			policy.StageCount = request.JetsonFabric.StageCount
		}
		if request.JetsonFabric.AllowColocatedStages {
			policy.AllowColocatedStages = true
		}
	}
	session, err := s.generations.Start(r.Context(), generationSpec{
		ModelID:   modelID,
		Prompt:    prompt,
		MaxTokens: chatMaxTokens(request),
		Policy:    policy,
	})
	if err != nil {
		writeGenerationStartError(w, err)
		return
	}
	defer session.Close()

	setGenerationHeaders(w, session.SessionID, session.Plan, session.Identity)
	if request.Stream {
		s.streamChatCompletion(
			w,
			r,
			session.RequestID,
			session.Model.ID,
			len(session.Plan.Stages),
			session.Body,
		)
		return
	}
	result, err := consumeGenerationEvents(session.Body, len(session.Plan.Stages), nil)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "server_error", "runtime_generation_failed", nil, err.Error())
		return
	}
	w.Header().Set("X-JetsonFabric-Stage-Calls", fmt.Sprintf("%d", result.StageCalls))
	w.Header().Set("X-JetsonFabric-Remote-Stage-Calls", fmt.Sprintf("%d", result.RemoteStageCalls))
	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:      session.RequestID,
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   session.Model.ID,
		Choices: []chatCompletionChoice{{
			Index: 0,
			Message: chatCompletionMessage{
				Role:    "assistant",
				Content: result.GeneratedText,
			},
			FinishReason: result.FinishReason,
		}},
		Usage: chatCompletionUsage{
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens,
			TotalTokens:      result.PromptTokens + result.CompletionTokens,
		},
	})
}

func writeGenerationStartError(w http.ResponseWriter, err error) {
	var startError *generationStartError
	if !errors.As(err, &startError) {
		writeInferenceAdmissionError(w, err, true)
		return
	}
	switch startError.kind {
	case generationUnknownModel:
		param := "model"
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", &param, err.Error())
	case generationRuntimeUnavailable:
		writeOpenAIError(w, http.StatusBadGateway, "server_error", string(startError.kind), nil, err.Error())
	default:
		writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", string(startError.kind), nil, err.Error())
	}
}

func chatMaxTokens(request chatCompletionRequest) int {
	if request.MaxCompletionTokens > 0 {
		return request.MaxCompletionTokens
	}
	return request.MaxTokens
}

func renderChatPrompt(messages []chatCompletionMessage) string {
	var prompt strings.Builder
	for _, message := range messages {
		content := strings.TrimSpace(message.Content)
		if content == "" {
			continue
		}
		role := strings.TrimSpace(message.Role)
		if role == "" {
			role = "user"
		}
		prompt.WriteString("<|im_start|>")
		prompt.WriteString(role)
		prompt.WriteString("\n")
		prompt.WriteString(content)
		prompt.WriteString("<|im_end|>\n")
	}
	if prompt.Len() == 0 {
		return ""
	}
	prompt.WriteString("<|im_start|>assistant\n")
	return prompt.String()
}

func writeOpenAIError(w http.ResponseWriter, status int, errorType string, code string, param *string, message string) {
	writeJSON(w, status, openAIErrorEnvelope{Error: openAIError{
		Message: message,
		Type:    errorType,
		Param:   param,
		Code:    code,
	}})
}
