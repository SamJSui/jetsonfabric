package coordinator

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/stageexec"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

type layerSplitRunRequest struct {
	RequestID            string `json:"request_id,omitempty"`
	Model                string `json:"model"`
	Payload              string `json:"payload"`
	MaxTokens            int    `json:"max_tokens,omitempty"`
	StageCount           *int   `json:"stage_count,omitempty"`
	AllowColocatedStages bool   `json:"allow_colocated_stages,omitempty"`
}

type layerSplitRunResponse struct {
	InterStagePayloadKind stagewire.PayloadKind    `json:"inter_stage_payload_kind"`
	Plan                  clusterplan.RoutePreview `json:"plan"`
	Result                stageexec.Result         `json:"result"`
}

type layerSplitRunErrorResponse struct {
	Error                 string                    `json:"error"`
	Message               string                    `json:"message"`
	InterStagePayloadKind stagewire.PayloadKind     `json:"inter_stage_payload_kind,omitempty"`
	Plan                  *clusterplan.RoutePreview `json:"plan,omitempty"`
	Result                *stageexec.Result         `json:"result,omitempty"`
}

func (s *Server) handleLayerSplitRun(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var runReq layerSplitRunRequest
	if err := json.NewDecoder(r.Body).Decode(&runReq); err != nil {
		writeLayerSplitRunError(w, http.StatusBadRequest, errorInvalidJSON, err.Error(), nil, nil)
		return
	}

	modelID := strings.TrimSpace(runReq.Model)
	if modelID == "" {
		writeLayerSplitRunError(w, http.StatusBadRequest, errorMissingModel, "model is required", nil, nil)
		return
	}
	if strings.TrimSpace(runReq.Payload) == "" {
		writeLayerSplitRunError(w, http.StatusBadRequest, errorMissingPayload, "payload is required", nil, nil)
		return
	}
	if runReq.StageCount != nil && *runReq.StageCount <= 0 {
		writeLayerSplitRunError(w, http.StatusBadRequest, errorInvalidStageCount, "stage_count must be greater than zero", nil, nil)
		return
	}
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeLayerSplitRunError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", modelID), nil, nil)
		return
	}
	if s.memberSource == nil {
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, "membership source is required for layer split run", nil, nil)
		return
	}

	policy := s.routePreviewPolicy(r)
	if runReq.AllowColocatedStages {
		policy.AllowColocatedStages = true
	}
	if runReq.StageCount != nil {
		policy.StageCount = *runReq.StageCount
	}
	plan := clusterplan.Preview(clusterplan.Request{
		Model: model, Members: s.memberSource.List(), Now: s.now(),
		StaleAfter: s.memberStaleAfter, Policy: policy,
	})
	if !plan.Valid {
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, fmt.Sprintf("no valid layer split route: %s", plan.Reason), &plan, nil)
		return
	}
	if plan.Mode != cluster.ExecutionModePipelineParallel || plan.StageCount < 2 {
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, "layer split run requires a pipeline_parallel plan with at least two stages", &plan, nil)
		return
	}

	result, err := stageexec.New(stageexec.Config{}).Generate(r.Context(), stageexec.Request{
		RequestID: runReq.RequestID,
		Model:     model.ID,
		Payload:   runReq.Payload,
		MaxTokens: runReq.MaxTokens,
		Plan:      plan,
	})
	if err != nil {
		writeLayerSplitRunError(w, http.StatusBadGateway, errorPipelineStageFailed, err.Error(), &plan, &result)
		return
	}
	writeJSON(w, http.StatusOK, newLayerSplitRunResponse(plan, result))
}

func newLayerSplitRunResponse(plan clusterplan.RoutePreview, result stageexec.Result) layerSplitRunResponse {
	return layerSplitRunResponse{
		InterStagePayloadKind: interStagePayloadKind(result),
		Plan:                  plan,
		Result:                result,
	}
}

func interStagePayloadKind(result stageexec.Result) stagewire.PayloadKind {
	if len(result.Stages) > 1 && result.Stages[0].PayloadKindOut != "" {
		return result.Stages[0].PayloadKindOut
	}
	return result.PayloadKind
}

func writeLayerSplitRunError(w http.ResponseWriter, status int, code errorCode, message string, plan *clusterplan.RoutePreview, result *stageexec.Result) {
	payloadKind := stagewire.PayloadKind("")
	if result != nil {
		payloadKind = interStagePayloadKind(*result)
	}
	writeJSON(w, status, layerSplitRunErrorResponse{
		Error: string(code), Message: message, InterStagePayloadKind: payloadKind,
		Plan: plan, Result: result,
	})
}
