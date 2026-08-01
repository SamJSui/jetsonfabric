package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

type deploymentSwitchRequest struct {
	DeploymentID         string `json:"deployment_id,omitempty"`
	Model                string `json:"model"`
	StageCount           int    `json:"stage_count,omitempty"`
	StageLayerCounts     []int  `json:"stage_layer_counts,omitempty"`
	AllowColocatedStages bool   `json:"allow_colocated_stages,omitempty"`
	ContextSize          int    `json:"ctx_size,omitempty"`
	Threads              int    `json:"threads,omitempty"`
	NGPULayers           *int   `json:"n_gpu_layers,omitempty"`
}

func (r deploymentSwitchRequest) spec() deploymentSpec {
	return deploymentSpec{
		DeploymentID:         r.DeploymentID,
		ModelID:              r.Model,
		StageCount:           r.StageCount,
		StageLayerCounts:     append([]int(nil), r.StageLayerCounts...),
		AllowColocatedStages: r.AllowColocatedStages,
		ContextSize:          r.ContextSize,
		Threads:              r.Threads,
		NGPULayers:           r.NGPULayers,
	}
}

type deploymentPlanResponse struct {
	DeploymentID string                              `json:"deployment_id"`
	Epoch        uint64                              `json:"epoch"`
	Model        clusterplan.DeploymentModelIdentity `json:"model"`
	Stages       []clusterplan.Stage                 `json:"stages"`
}

type deploymentStatusResponse struct {
	Phase           deploymentPhase          `json:"phase"`
	InFlight        int                      `json:"in_flight"`
	InFlightByEpoch map[uint64]int           `json:"in_flight_by_epoch,omitempty"`
	LastError       string                   `json:"last_error,omitempty"`
	Active          *deploymentPlanResponse  `json:"active"`
	Preparing       *deploymentPlanResponse  `json:"preparing,omitempty"`
	Draining        []deploymentPlanResponse `json:"draining,omitempty"`
}

type deploymentSwitchResponse struct {
	Phase         deploymentPhase                     `json:"phase"`
	Active        deploymentPlanResponse              `json:"active"`
	Compatibility clusterplan.DeploymentCompatibility `json:"compatibility"`
	CleanupError  string                              `json:"cleanup_error,omitempty"`
}

func (s *Server) handleDeploymentStatus(w http.ResponseWriter, _ *http.Request) {
	snapshot := s.deployments.snapshot()
	response := deploymentStatusResponse{
		Phase:           snapshot.Phase,
		InFlight:        snapshot.InFlight,
		InFlightByEpoch: snapshot.InFlightByEpoch,
		LastError:       snapshot.LastError,
	}
	if snapshot.Active != nil {
		active := newDeploymentPlanResponse(*snapshot.Active)
		response.Active = &active
	}
	if snapshot.Preparing != nil {
		preparing := newDeploymentPlanResponse(*snapshot.Preparing)
		response.Preparing = &preparing
	}
	for _, plan := range snapshot.Draining {
		response.Draining = append(response.Draining, newDeploymentPlanResponse(plan))
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleDeploymentSwitch(w http.ResponseWriter, r *http.Request) {
	request, ok := decodeDeploymentSwitchRequest(w, r)
	if !ok {
		return
	}
	if _, exists := s.registry.Find(request.Model); !exists {
		writeError(w, http.StatusNotFound, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", request.Model))
		return
	}
	if s.memberSource == nil {
		writeError(w, http.StatusServiceUnavailable, errorDeploymentUnavailable, "membership source is required for deployment switching")
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), s.deployments.transitionTimeout)
	defer cancel()
	build, cleanupErr, err := s.deployments.switchDeployment(ctx, request.spec(), true)
	if err != nil {
		writeDeploymentSwitchError(w, err)
		return
	}
	snapshot := s.deployments.snapshot()
	response := deploymentSwitchResponse{
		Phase:         snapshot.Phase,
		Active:        newDeploymentPlanResponse(build.result.Plan),
		Compatibility: build.result.Compatibility,
	}
	if cleanupErr != nil {
		response.CleanupError = cleanupErr.Error()
	}
	writeJSON(w, http.StatusOK, response)
}

func decodeDeploymentSwitchRequest(w http.ResponseWriter, r *http.Request) (deploymentSwitchRequest, bool) {
	defer r.Body.Close()
	var request deploymentSwitchRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, err.Error())
		return deploymentSwitchRequest{}, false
	}
	request.Model = strings.TrimSpace(request.Model)
	request.DeploymentID = strings.TrimSpace(request.DeploymentID)
	switch {
	case request.Model == "":
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
	case request.StageCount < 0:
		writeError(w, http.StatusBadRequest, errorInvalidStageCount, "stage_count cannot be negative")
	case hasNonPositiveValue(request.StageLayerCounts):
		writeError(w, http.StatusBadRequest, errorInvalidStageCount, "stage_layer_counts must contain only positive values")
	case request.ContextSize < 0 || request.Threads < 0:
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "ctx_size and threads cannot be negative")
	case request.NGPULayers != nil && *request.NGPULayers < 0:
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "n_gpu_layers cannot be negative")
	default:
		return request, true
	}
	return deploymentSwitchRequest{}, false
}

func hasNonPositiveValue(values []int) bool {
	for _, value := range values {
		if value <= 0 {
			return true
		}
	}
	return false
}

func writeDeploymentSwitchError(w http.ResponseWriter, err error) {
	status := http.StatusBadGateway
	code := errorDeploymentSwitchFailed
	switch {
	case errors.Is(err, errDeploymentTransitioning):
		status = http.StatusConflict
		code = errorDeploymentTransitioning
	case errors.Is(err, errDeploymentPlanInvalid):
		status = http.StatusConflict
		code = errorDeploymentPlanInvalid
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		status = http.StatusRequestTimeout
	case errors.Is(err, errDeploymentUnavailable):
		status = http.StatusServiceUnavailable
	}
	writeError(w, status, code, err.Error())
}

func newDeploymentPlanResponse(plan clusterplan.DeploymentPlan) deploymentPlanResponse {
	return deploymentPlanResponse{
		DeploymentID: plan.Identity().DeploymentID,
		Epoch:        plan.Identity().Epoch,
		Model:        plan.Model(),
		Stages:       plan.Stages(),
	}
}
