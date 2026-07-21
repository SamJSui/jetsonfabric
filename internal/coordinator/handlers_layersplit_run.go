package coordinator

import (
	"encoding/json"
	"errors"
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
	RuntimeIdentity       pipelineRuntimeIdentity  `json:"runtime_identity"`
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
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, "membership source is required for pipeline run", nil, nil)
		return
	}

	admission, err := s.deployments.admit(modelID)
	if err != nil {
		writeInferenceAdmissionError(w, err, false)
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
		if runReq.AllowColocatedStages {
			policy.AllowColocatedStages = true
		}
		if runReq.StageCount != nil {
			policy.StageCount = *runReq.StageCount
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
			writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, err.Error(), nil, nil)
			return
		}
		identity = legacyIdentity
		plan = clusterplan.PreviewPipeline(clusterplan.Request{
			Model: model, Members: members, Now: s.now(),
			StaleAfter: s.memberStaleAfter, Policy: policy,
		})
	}
	if !plan.Valid {
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, fmt.Sprintf("no valid pipeline route: %s", plan.Reason), &plan, nil)
		return
	}
	if plan.Mode != cluster.ExecutionModePipelineParallel || plan.StageCount < 1 {
		writeLayerSplitRunError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, "pipeline run requires a pipeline_parallel plan with at least one stage", &plan, nil)
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
	writeJSON(w, http.StatusOK, newLayerSplitRunResponse(identity, plan, result))
}

func runtimeIdentityForDeployment(plan clusterplan.DeploymentPlan) pipelineRuntimeIdentity {
	model := plan.Model()
	return pipelineRuntimeIdentity{
		Engine:        model.Engine,
		ModelID:       model.ModelID,
		ModelSHA256:   model.ModelSHA256,
		ExecutionMode: model.ExecutionMode,
		DeploymentID:  plan.Identity().DeploymentID,
		Epoch:         plan.Identity().Epoch,
	}
}

func writeInferenceAdmissionError(w http.ResponseWriter, err error, openAI bool) {
	status := http.StatusServiceUnavailable
	code := errorDeploymentTransitioning
	if errors.Is(err, errModelNotActive) {
		status = http.StatusConflict
		code = errorModelNotActive
	}
	if openAI {
		writeOpenAIError(w, status, "server_error", string(code), nil, err.Error())
		return
	}
	writeLayerSplitRunError(w, status, code, err.Error(), nil, nil)
}

func newLayerSplitRunResponse(identity pipelineRuntimeIdentity, plan clusterplan.RoutePreview, result stageexec.Result) layerSplitRunResponse {
	return layerSplitRunResponse{
		InterStagePayloadKind: interStagePayloadKind(result),
		RuntimeIdentity:       identity,
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
