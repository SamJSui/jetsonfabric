package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
)

func (s *Server) handleLayerSplitPlan(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSpace(r.URL.Query().Get("model"))
	if modelID == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", modelID))
		return
	}
	if !slices.Contains(model.PlacementModes, cluster.ExecutionModePipelineParallel) {
		writeError(w, http.StatusBadRequest, errorLayerSplitUnsupported, fmt.Sprintf("model %q does not allow layer_split placement", model.ID))
		return
	}

	candidates := s.layerSplitCandidates(model)
	plan, err := layersplit.PlanForModel(model, candidates)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoLayerSplitRoute, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, plan)
}

func (s *Server) handleLayerSplitCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req chat.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "messages must contain at least one message")
		return
	}

	model, ok := s.registry.Find(req.Model)
	if !ok {
		writeError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", req.Model))
		return
	}
	if !slices.Contains(model.PlacementModes, cluster.ExecutionModePipelineParallel) {
		writeError(w, http.StatusBadRequest, errorLayerSplitUnsupported, fmt.Sprintf("model %q does not allow layer_split placement", model.ID))
		return
	}

	candidates := s.layerSplitCandidates(model)
	plan, err := layersplit.PlanForModel(model, candidates)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoLayerSplitRoute, err.Error())
		return
	}

	start := time.Now()
	sessionID := fmt.Sprintf("layer-session-%d", s.now().UnixNano())
	requestID := fmt.Sprintf("layer-request-%d", s.now().UnixNano())
	payload := lastMessageContent(req.Messages)
	if strings.TrimSpace(payload) == "" {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "last message content is required")
		return
	}

	stageResponses := make([]layersplit.ActivationResponse, 0, len(plan.Stages))
	for _, stage := range plan.Stages {
		stageReq := layersplit.BuildStageRequest(sessionID, requestID, model.ID, stage, payload, s.layerTransportKind)
		resp, err := s.layerTransport.RunStage(r.Context(), layersplit.StageTarget{
			NodeName: stage.NodeName,
			BaseURL:  stage.BaseURL,
			Stage:    stage,
		}, stageReq)
		if err != nil {
			writeError(w, http.StatusBadGateway, errorLayerSplitStageFailed, err.Error())
			return
		}
		stageResponses = append(stageResponses, resp)
		payload = resp.Payload
	}

	latency := time.Since(start)
	content := fmt.Sprintf("synthetic layer_split response: %s", payload)
	outputTokens := len(strings.Fields(content))
	resp := chat.CompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model.ID,
		Choices: []chat.Choice{
			{
				Index: 0,
				Message: chat.Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: &chat.Usage{
			PromptTokens:     len(strings.Fields(lastMessageContent(req.Messages))),
			CompletionTokens: outputTokens,
			TotalTokens:      len(strings.Fields(lastMessageContent(req.Messages))) + outputTokens,
		},
		Route: s.layerSplitRouteMetadata(plan, stageResponses, latency),
	}
	if err := s.benchmarkRecorder.Record(r.Context(), benchmarks.Record{
		Timestamp:    s.now(),
		ModelID:      model.ID,
		NodeName:     strings.Join(stageNodeNames(stageResponses), ","),
		RouteMode:    cluster.RouteModeLayerSplit,
		BackendID:    "layer-split",
		BackendKind:  cluster.RuntimeKindLlamaCPP,
		LatencyMS:    latency.Milliseconds(),
		OutputTokens: outputTokens,
	}); err != nil {
		writeError(w, http.StatusInternalServerError, errorBenchmarkRecordFailed, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}
