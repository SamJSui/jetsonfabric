package coordinator

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
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
	model, ok := s.registry.Find(modelID)
	if !ok {
		param := "model"
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", &param, fmt.Sprintf("model %q is not in the JetsonFabric registry", modelID))
		return
	}
	prompt := renderChatPrompt(request.Messages)
	if prompt == "" {
		param := "messages"
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "messages_required", &param, "at least one non-empty message is required")
		return
	}
	if s.memberSource == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "membership_unavailable", nil, "membership source is required for pipeline chat completion")
		return
	}

	admission, err := s.deployments.admit(modelID)
	if err != nil {
		writeInferenceAdmissionError(w, err, true)
		return
	}
	defer admission.Release()

	var plan clusterplan.RoutePreview
	var identity pipelineRuntimeIdentity
	if admission.Plan != nil {
		plan = admission.Plan.RoutePreview()
		identity = runtimeIdentityForDeployment(*admission.Plan)
	} else {
		policy := s.routePreviewPolicy(r)
		if request.JetsonFabric != nil {
			if request.JetsonFabric.StageCount > 0 {
				policy.StageCount = request.JetsonFabric.StageCount
			}
			if request.JetsonFabric.AllowColocatedStages {
				policy.AllowColocatedStages = true
			}
		}
		requiredStages := policy.StageCount
		if requiredStages <= 0 {
			requiredStages = 1
			policy.StageCount = requiredStages
		}
		members, legacyIdentity, err := selectPipelineRuntimeMembers(
			model,
			s.memberSource.List(),
			s.now(),
			s.memberStaleAfter,
			requiredStages,
		)
		if err != nil {
			writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "runtime_identity_unavailable", nil, err.Error())
			return
		}
		identity = legacyIdentity
		plan = clusterplan.PreviewPipeline(clusterplan.Request{
			Model: model, Members: members, Now: s.now(),
			StaleAfter: s.memberStaleAfter, Policy: policy,
		})
	}
	if !plan.Valid || plan.Mode != cluster.ExecutionModePipelineParallel || plan.StageCount < 1 {
		writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "pipeline_route_unavailable", nil, fmt.Sprintf("no valid pipeline route for model %q: %s", modelID, plan.Reason))
		return
	}

	requestID := fmt.Sprintf("chatcmpl-%d", s.now().UnixNano())
	sessionID := fmt.Sprintf("session-%d", s.now().UnixNano())
	generationRequest := runtimebridge.GenerationRequest{
		RequestID: requestID,
		SessionID: sessionID,
		ModelID:   model.ID,
		Prompt:    prompt,
		MaxTokens: chatMaxTokens(request),
		Stages:    plan.Stages,
	}
	if identity.DeploymentID != "" {
		generationRequest.Deployment = &runtimebridge.DeploymentIdentity{
			DeploymentID: identity.DeploymentID,
			Epoch:        identity.Epoch,
			ModelID:      identity.ModelID,
			ModelSHA256:  identity.ModelSHA256,
		}
	}
	stream, err := s.generationClient.Start(r.Context(), plan.Stages[0].APIURL, generationRequest)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "server_error", "runtime_generation_failed", nil, err.Error())
		return
	}
	defer stream.Body.Close()

	setGenerationHeaders(w, sessionID, plan, identity)
	if request.Stream {
		s.streamChatCompletion(w, r, requestID, model.ID, len(plan.Stages), stream.Body)
		return
	}
	result, err := consumeGenerationEvents(stream.Body, len(plan.Stages), nil)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "server_error", "runtime_generation_failed", nil, err.Error())
		return
	}
	w.Header().Set("X-JetsonFabric-Stage-Calls", fmt.Sprintf("%d", result.StageCalls))
	w.Header().Set("X-JetsonFabric-Remote-Stage-Calls", fmt.Sprintf("%d", result.RemoteStageCalls))
	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model.ID,
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
