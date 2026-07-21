package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

const (
	defaultDeploymentContextSize = 4096
	deploymentSwitchTimeout      = 30 * time.Minute
	deploymentCleanupTimeout     = 10 * time.Minute
)

type deploymentSwitchRequest struct {
	DeploymentID         string `json:"deployment_id,omitempty"`
	Model                string `json:"model"`
	StageCount           int    `json:"stage_count,omitempty"`
	AllowColocatedStages bool   `json:"allow_colocated_stages,omitempty"`
	ContextSize          int    `json:"ctx_size,omitempty"`
	Threads              int    `json:"threads,omitempty"`
	NGPULayers           *int   `json:"n_gpu_layers,omitempty"`
}

type deploymentPlanResponse struct {
	DeploymentID string                              `json:"deployment_id"`
	Epoch        uint64                              `json:"epoch"`
	Model        clusterplan.DeploymentModelIdentity `json:"model"`
	Stages       []clusterplan.Stage                 `json:"stages"`
}

type deploymentStatusResponse struct {
	Phase     deploymentPhase         `json:"phase"`
	InFlight  int                     `json:"in_flight"`
	LastError string                  `json:"last_error,omitempty"`
	Active    *deploymentPlanResponse `json:"active"`
	Recovery  *deploymentPlanResponse `json:"recovery,omitempty"`
}

type deploymentSwitchResponse struct {
	Phase         deploymentPhase                     `json:"phase"`
	Active        deploymentPlanResponse              `json:"active"`
	Compatibility clusterplan.DeploymentCompatibility `json:"compatibility"`
}

func (s *Server) handleDeploymentStatus(w http.ResponseWriter, _ *http.Request) {
	snapshot := s.deployments.snapshot()
	response := deploymentStatusResponse{
		Phase:     snapshot.Phase,
		InFlight:  snapshot.InFlight,
		LastError: snapshot.LastError,
	}
	if snapshot.Active != nil {
		active := newDeploymentPlanResponse(*snapshot.Active)
		response.Active = &active
	}
	if snapshot.Recovery != nil {
		recovery := newDeploymentPlanResponse(*snapshot.Recovery)
		response.Recovery = &recovery
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleDeploymentSwitch(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var request deploymentSwitchRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, err.Error())
		return
	}
	request.Model = strings.TrimSpace(request.Model)
	request.DeploymentID = strings.TrimSpace(request.DeploymentID)
	if request.Model == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	if request.StageCount < 0 {
		writeError(w, http.StatusBadRequest, errorInvalidStageCount, "stage_count must be greater than zero")
		return
	}
	if request.ContextSize < 0 || request.Threads < 0 {
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "ctx_size and threads cannot be negative")
		return
	}
	if request.NGPULayers != nil && *request.NGPULayers < 0 {
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "n_gpu_layers cannot be negative")
		return
	}
	model, ok := s.registry.Find(request.Model)
	if !ok {
		writeError(w, http.StatusNotFound, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", request.Model))
		return
	}
	if s.memberSource == nil {
		writeError(w, http.StatusServiceUnavailable, errorDeploymentUnavailable, "membership source is required for deployment switching")
		return
	}

	snapshot := s.deployments.snapshot()
	identity := clusterplan.DeploymentIdentity{
		DeploymentID: request.DeploymentID,
		Epoch:        snapshot.ProposedEpoch,
	}
	if identity.DeploymentID == "" {
		identity.DeploymentID = fmt.Sprintf("deployment-%d-%d", identity.Epoch, s.now().UnixNano())
	}
	policy := s.clusterPlanPolicy
	if request.StageCount > 0 {
		policy.StageCount = request.StageCount
	}
	if request.AllowColocatedStages {
		policy.AllowColocatedStages = true
	}
	members := append([]membership.Member(nil), s.memberSource.List()...)
	build, err := clusterplan.BuildDeploymentPlan(clusterplan.DeploymentBuildRequest{
		Identity:   identity,
		Model:      model,
		Members:    members,
		Now:        s.now(),
		StaleAfter: s.memberStaleAfter,
		Policy:     policy,
	})
	if err != nil {
		writeError(w, http.StatusConflict, errorDeploymentPlanInvalid, err.Error())
		return
	}

	transitionCtx, cancelTransition := context.WithTimeout(context.Background(), deploymentSwitchTimeout)
	defer cancelTransition()
	previous, err := s.deployments.beginTransition(transitionCtx, identity.Epoch)
	if err != nil {
		status := http.StatusConflict
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			status = http.StatusRequestTimeout
		}
		writeError(w, status, errorDeploymentTransitioning, err.Error())
		return
	}
	if err := s.applyDeploymentSwitch(transitionCtx, previous, build.Plan, model, members, request); err != nil {
		s.deployments.fail(previous, err)
		writeError(w, http.StatusBadGateway, errorDeploymentSwitchFailed, err.Error())
		return
	}
	s.deployments.publish(build.Plan)
	writeJSON(w, http.StatusOK, deploymentSwitchResponse{
		Phase:         deploymentPhaseActive,
		Active:        newDeploymentPlanResponse(build.Plan),
		Compatibility: build.Compatibility,
	})
}

func (s *Server) applyDeploymentSwitch(
	ctx context.Context,
	previous *clusterplan.DeploymentPlan,
	next clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	members []membership.Member,
	request deploymentSwitchRequest,
) error {
	if err := s.ensureIncomingPlanRuntimesIdle(ctx, previous, next); err != nil {
		return err
	}
	if previous != nil {
		if err := s.unloadPlan(ctx, *previous); err != nil {
			return fmt.Errorf("unload active deployment %q: %w", previous.Identity().DeploymentID, err)
		}
	}
	if err := s.ensurePlanRuntimesIdle(ctx, next); err != nil {
		return err
	}
	if err := s.loadPlan(ctx, next, model, members, request); err != nil {
		return s.cleanupFailedPlan(next, err)
	}
	if err := s.waitForPlanState(ctx, next, "ready", false); err != nil {
		return s.cleanupFailedPlan(next, fmt.Errorf("ready barrier: %w", err))
	}
	if err := s.activatePlan(ctx, next); err != nil {
		return s.cleanupFailedPlan(next, err)
	}
	if err := s.waitForPlanState(ctx, next, "active", true); err != nil {
		return s.cleanupFailedPlan(next, fmt.Errorf("active barrier: %w", err))
	}
	return nil
}

func (s *Server) cleanupFailedPlan(plan clusterplan.DeploymentPlan, cause error) error {
	ctx, cancel := context.WithTimeout(context.Background(), deploymentCleanupTimeout)
	defer cancel()
	if err := s.unloadPlan(ctx, plan); err != nil {
		return fmt.Errorf("%w; cleanup deployment %q: %v", cause, plan.Identity().DeploymentID, err)
	}
	return cause
}

func (s *Server) ensureIncomingPlanRuntimesIdle(
	ctx context.Context,
	previous *clusterplan.DeploymentPlan,
	next clusterplan.DeploymentPlan,
) error {
	previousNodes := make(map[string]struct{})
	if previous != nil {
		for _, stage := range previous.Stages() {
			previousNodes[stage.NodeID] = struct{}{}
		}
	}
	for _, stage := range next.Stages() {
		if _, reused := previousNodes[stage.NodeID]; reused {
			continue
		}
		if err := s.ensureRuntimeIdle(ctx, stage); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) ensurePlanRuntimesIdle(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	for _, stage := range plan.Stages() {
		if err := s.ensureRuntimeIdle(ctx, stage); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) ensureRuntimeIdle(ctx context.Context, stage clusterplan.Stage) error {
	status, err := s.deploymentClient.Status(ctx, stage.APIURL)
	if err != nil {
		return fmt.Errorf("read runtime status for node %q: %w", stage.NodeID, err)
	}
	if status.Resident {
		return fmt.Errorf("runtime on node %q is not idle; resident deployment must be coordinator-managed before replacement", stage.NodeID)
	}
	return nil
}

func (s *Server) loadPlan(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	members []membership.Member,
	request deploymentSwitchRequest,
) error {
	byNode := make(map[string]membership.Member, len(members))
	for _, member := range members {
		byNode[member.NodeID] = member
	}
	ctxSize := request.ContextSize
	if ctxSize == 0 {
		ctxSize = defaultDeploymentContextSize
	}
	for _, stage := range plan.Stages() {
		member, ok := byNode[stage.NodeID]
		if !ok {
			return fmt.Errorf("deployment member %q disappeared from the immutable snapshot", stage.NodeID)
		}
		backend := cluster.ComputeBackend(capabilityString(member.Capabilities, cluster.CapabilityRuntimeComputeBackend))
		nGPULayers := 0
		if backend == cluster.ComputeBackendCUDA {
			nGPULayers = 999
		}
		if request.NGPULayers != nil {
			nGPULayers = *request.NGPULayers
		}
		response, err := s.deploymentClient.Load(ctx, stage.APIURL, runtimebridge.LoadDeploymentRequest{
			DeploymentID:   plan.Identity().DeploymentID,
			ModelID:        model.ID,
			Engine:         string(plan.Model().Engine),
			ComputeBackend: string(backend),
			ModelPath:      model.ArtifactPath,
			CtxSize:        ctxSize,
			NGPULayers:     nGPULayers,
			Threads:        request.Threads,
			Mode:           string(plan.Model().ExecutionMode),
			StageIndex:     stage.StageIndex,
			StageCount:     stage.StageCount,
			LayerStart:     stage.LayerStart,
			LayerEnd:       stage.LayerEnd,
		})
		if err != nil {
			return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
		}
		if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "ready", false); err != nil {
			return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
		}
	}
	return nil
}

func (s *Server) activatePlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	for _, stage := range plan.Stages() {
		response, err := s.deploymentClient.Activate(ctx, stage.APIURL, plan.Identity().DeploymentID)
		if err != nil {
			return fmt.Errorf("activate deployment on node %q: %w", stage.NodeID, err)
		}
		if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "active", true); err != nil {
			return fmt.Errorf("activate deployment on node %q: %w", stage.NodeID, err)
		}
	}
	return nil
}

func (s *Server) unloadPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	stages := plan.Stages()
	toUnload := make([]clusterplan.Stage, 0, len(stages))
	var failures []string
	for _, stage := range stages {
		status, err := s.deploymentClient.Status(ctx, stage.APIURL)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q status: %v", stage.NodeID, err))
			continue
		}
		if !status.Resident && status.State == "idle" {
			continue
		}
		if status.Deployment == nil || status.Deployment.DeploymentID != plan.Identity().DeploymentID {
			failures = append(failures, fmt.Sprintf("node %q hosts a different or unidentified deployment", stage.NodeID))
			continue
		}
		toUnload = append(toUnload, stage)
	}
	if len(failures) > 0 {
		return errors.New(strings.Join(failures, "; "))
	}
	for _, stage := range toUnload {
		response, err := s.deploymentClient.Unload(ctx, stage.APIURL, plan.Identity().DeploymentID)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q: %v", stage.NodeID, err))
			continue
		}
		if response.Resident || response.Active || response.State != "idle" {
			failures = append(failures, fmt.Sprintf("node %q did not return to idle", stage.NodeID))
		}
	}
	if len(failures) > 0 {
		sort.Strings(failures)
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

func (s *Server) waitForPlanState(ctx context.Context, plan clusterplan.DeploymentPlan, state string, active bool) error {
	for _, stage := range plan.Stages() {
		status, err := s.deploymentClient.Status(ctx, stage.APIURL)
		if err != nil {
			return fmt.Errorf("read runtime status for node %q: %w", stage.NodeID, err)
		}
		if err := validateRuntimeStatus(status, plan, stage, state, active); err != nil {
			return fmt.Errorf("node %q: %w", stage.NodeID, err)
		}
	}
	return nil
}

func validateRuntimeStatus(
	status runtimebridge.DeploymentStatus,
	plan clusterplan.DeploymentPlan,
	stage clusterplan.Stage,
	state string,
	active bool,
) error {
	if !status.Resident || status.Active != active || status.State != state || status.Deployment == nil {
		return fmt.Errorf("unexpected runtime status resident=%t active=%t state=%q", status.Resident, status.Active, status.State)
	}
	if status.Deployment.DeploymentID != plan.Identity().DeploymentID || status.Deployment.ModelID != plan.Model().ModelID {
		return fmt.Errorf("runtime reports deployment %q model %q, want deployment %q model %q for stage %d", status.Deployment.DeploymentID, status.Deployment.ModelID, plan.Identity().DeploymentID, plan.Model().ModelID, stage.StageIndex)
	}
	if err := validateRuntimeModelMemory(status, plan, stage, active); err != nil {
		return fmt.Errorf("stage %d model residency: %w", stage.StageIndex, err)
	}
	return nil
}

func validateRuntimeModelMemory(
	status runtimebridge.DeploymentStatus,
	plan clusterplan.DeploymentPlan,
	stage clusterplan.Stage,
	active bool,
) error {
	memory := status.ModelMemory
	if memory == nil {
		return errors.New("runtime omitted model_memory")
	}
	model := plan.Model()
	if memory.LayerStart != stage.LayerStart || memory.LayerEnd != stage.LayerEnd {
		return fmt.Errorf("reported layers [%d,%d), want [%d,%d)", memory.LayerStart, memory.LayerEnd, stage.LayerStart, stage.LayerEnd)
	}
	if memory.LayerCount != model.LayerCount {
		return fmt.Errorf("reported layer_count %d, want %d", memory.LayerCount, model.LayerCount)
	}
	if memory.ResidentWeightBytes == 0 || memory.TotalWeightBytes == 0 || memory.ResidentTensorCount == 0 {
		return errors.New("runtime reported empty model residency")
	}
	partitioned := stage.LayerStart != 0 || stage.LayerEnd != model.LayerCount
	if memory.Partitioned != partitioned {
		return fmt.Errorf("reported partitioned=%t, want %t", memory.Partitioned, partitioned)
	}
	if !partitioned && memory.ResidentWeightBytes != memory.TotalWeightBytes {
		return fmt.Errorf("full-range stage retains %d bytes from a %d-byte model", memory.ResidentWeightBytes, memory.TotalWeightBytes)
	}
	if partitioned && memory.ResidentWeightBytes >= memory.TotalWeightBytes {
		return fmt.Errorf("partition retains %d bytes from a %d-byte model", memory.ResidentWeightBytes, memory.TotalWeightBytes)
	}
	if memory.Pinned != active {
		return fmt.Errorf("reported pinned=%t, want %t", memory.Pinned, active)
	}
	return nil
}

func newDeploymentPlanResponse(plan clusterplan.DeploymentPlan) deploymentPlanResponse {
	return deploymentPlanResponse{
		DeploymentID: plan.Identity().DeploymentID,
		Epoch:        plan.Identity().Epoch,
		Model:        plan.Model(),
		Stages:       plan.Stages(),
	}
}
