package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
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

func (r deploymentSwitchRequest) spec() deploymentSpec {
	return deploymentSpec{
		DeploymentID:         r.DeploymentID,
		ModelID:              r.Model,
		StageCount:           r.StageCount,
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

type deploymentBuild struct {
	model   cluster.ModelProfile
	members []membership.Member
	policy  clusterplan.Policy
	result  clusterplan.DeploymentBuildResult
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
	case request.ContextSize < 0 || request.Threads < 0:
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "ctx_size and threads cannot be negative")
	case request.NGPULayers != nil && *request.NGPULayers < 0:
		writeError(w, http.StatusBadRequest, errorDeploymentConfigInvalid, "n_gpu_layers cannot be negative")
	default:
		return request, true
	}
	return deploymentSwitchRequest{}, false
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

func (c *DeploymentController) switchDeploymentLocked(
	ctx context.Context,
	spec deploymentSpec,
	force bool,
) (deploymentBuild, error, error) {
	build, err := c.buildDeployment(spec)
	if err != nil {
		return deploymentBuild{}, nil, fmt.Errorf("%w: %v", errDeploymentPlanInvalid, err)
	}
	current := c.state.snapshot()
	if !force && current.Active != nil && plansEquivalent(*current.Active, build.result.Plan) {
		return build, nil, nil
	}
	cleanupErr, err := c.transitionDeployment(ctx, build, spec)
	return build, cleanupErr, err
}

func (c *DeploymentController) buildDeployment(spec deploymentSpec) (deploymentBuild, error) {
	model, ok := c.registry.Find(spec.ModelID)
	if !ok {
		return deploymentBuild{}, fmt.Errorf("model %q is not in the registry", spec.ModelID)
	}
	if c.memberSource == nil {
		return deploymentBuild{}, errDeploymentUnavailable
	}
	snapshot := c.state.snapshot()
	identity := clusterplan.DeploymentIdentity{
		DeploymentID: spec.DeploymentID,
		Epoch:        snapshot.ProposedEpoch,
	}
	if identity.DeploymentID == "" {
		identity.DeploymentID = fmt.Sprintf("deployment-%d-%d", identity.Epoch, c.now().UnixNano())
	}
	policy := deploymentPolicy(c.planPolicy, spec)
	members := append([]membership.Member(nil), c.memberSource.List()...)
	result, err := clusterplan.BuildDeploymentPlan(clusterplan.DeploymentBuildRequest{
		Identity: identity, Model: model, Members: members,
		Now: c.now(), StaleAfter: c.memberStaleAfter, Policy: policy,
	})
	if err != nil {
		return deploymentBuild{}, err
	}
	return deploymentBuild{model: model, members: members, policy: policy, result: result}, nil
}

func deploymentPolicy(base clusterplan.Policy, spec deploymentSpec) clusterplan.Policy {
	policy := base
	if spec.StageCount > 0 {
		policy.StageCount = spec.StageCount
	}
	if spec.AllowColocatedStages {
		policy.AllowColocatedStages = true
	}
	return policy
}

func (c *DeploymentController) transitionDeployment(
	ctx context.Context,
	build deploymentBuild,
	spec deploymentSpec,
) (error, error) {
	previous, err := c.state.beginTransition(build.result.Plan)
	if err != nil {
		return nil, err
	}
	if err := c.preparePlan(ctx, build, spec); err != nil {
		failure := c.rollbackPreparedPlan(build.result.Plan, previous, err)
		return nil, failure
	}

	intent := intentFromSpec(spec, build.policy)
	previous = c.state.publish(build.result.Plan, intent)
	if previous == nil {
		return nil, nil
	}
	if err := c.retirePlan(ctx, *previous); err != nil {
		c.state.recordReconcileError(err, true)
		return err, nil
	}
	return nil, nil
}

func (c *DeploymentController) preparePlan(
	ctx context.Context,
	build deploymentBuild,
	spec deploymentSpec,
) error {
	if err := c.loadPlan(ctx, build.result.Plan, build.model, build.members, spec); err != nil {
		return fmt.Errorf("prepare deployment: %w", err)
	}
	if err := c.activatePlan(ctx, build.result.Plan); err != nil {
		return fmt.Errorf("activate deployment: %w", err)
	}
	return nil
}

func (c *DeploymentController) rollbackPreparedPlan(
	plan clusterplan.DeploymentPlan,
	previous *clusterplan.DeploymentPlan,
	cause error,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), c.cleanupTimeout)
	defer cancel()
	cleanupErr := c.cleanupPlan(ctx, plan)
	healthy := previous == nil || activePlanHealthy(*previous, c.members(), c.now(), c.memberStaleAfter)
	if cleanupErr != nil {
		cause = fmt.Errorf("%w; rollback cleanup: %v", cause, cleanupErr)
	}
	c.state.rollback(cause, healthy)
	return cause
}

func (c *DeploymentController) retirePlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	drainErr := c.drainPlan(ctx, plan)
	if err := c.state.waitForEpoch(ctx, plan.Identity().Epoch); err != nil {
		return errors.Join(
			drainErr,
			fmt.Errorf("wait for deployment %q sessions: %w", plan.Identity().DeploymentID, err),
		)
	}
	unloadErr := c.unloadPlan(ctx, plan)
	if drainErr != nil || unloadErr != nil {
		return errors.Join(
			wrappedPlanError("drain", plan, drainErr),
			wrappedPlanError("unload", plan, unloadErr),
		)
	}
	c.state.finishDraining(plan.Identity().Epoch)
	return nil
}

func wrappedPlanError(operation string, plan clusterplan.DeploymentPlan, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s deployment %q: %w", operation, plan.Identity().DeploymentID, err)
}

func (c *DeploymentController) cleanupPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	drainErr := c.drainPlan(ctx, plan)
	unloadErr := c.unloadPlan(ctx, plan)
	switch {
	case drainErr != nil && unloadErr != nil:
		return fmt.Errorf("drain: %v; unload: %v", drainErr, unloadErr)
	case drainErr != nil:
		return drainErr
	default:
		return unloadErr
	}
}

func (c *DeploymentController) loadPlan(
	ctx context.Context,
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	members []membership.Member,
	spec deploymentSpec,
) error {
	byNode := make(map[string]membership.Member, len(members))
	for _, member := range members {
		byNode[member.NodeID] = member
	}
	for _, stage := range plan.Stages() {
		member, ok := byNode[stage.NodeID]
		if !ok {
			return fmt.Errorf("deployment member %q disappeared from the immutable snapshot", stage.NodeID)
		}
		loadRequest := newRuntimeLoadRequest(plan, model, member, stage, spec)
		response, err := c.runtimeClient.Load(ctx, stage.APIURL, loadRequest)
		if err != nil {
			return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
		}
		if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "ready", false); err != nil {
			return fmt.Errorf("load deployment on node %q: %w", stage.NodeID, err)
		}
	}
	return nil
}

func newRuntimeLoadRequest(
	plan clusterplan.DeploymentPlan,
	model cluster.ModelProfile,
	member membership.Member,
	stage clusterplan.Stage,
	spec deploymentSpec,
) runtimebridge.LoadDeploymentRequest {
	ctxSize := spec.ContextSize
	if ctxSize == 0 {
		ctxSize = defaultDeploymentContextSize
	}
	backend := cluster.ComputeBackend(capabilityString(member.Capabilities, cluster.CapabilityRuntimeComputeBackend))
	nGPULayers := 0
	if backend == cluster.ComputeBackendCUDA {
		nGPULayers = 999
	}
	if spec.NGPULayers != nil {
		nGPULayers = *spec.NGPULayers
	}
	return runtimebridge.LoadDeploymentRequest{
		DeploymentID: plan.Identity().DeploymentID, Epoch: plan.Identity().Epoch,
		ModelID: model.ID, ModelSHA256: plan.Model().ModelSHA256,
		Engine: string(plan.Model().Engine), ComputeBackend: string(backend),
		ModelPath: model.ArtifactPath, CtxSize: ctxSize, NGPULayers: nGPULayers,
		Threads: spec.Threads, Mode: string(plan.Model().ExecutionMode),
		StageIndex: stage.StageIndex, StageCount: stage.StageCount,
		LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
	}
}

func (c *DeploymentController) activatePlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	identity := runtimeDeploymentIdentity(plan)
	for _, stage := range plan.Stages() {
		response, err := c.runtimeClient.Activate(ctx, stage.APIURL, identity)
		if err != nil {
			return fmt.Errorf("node %q: %w", stage.NodeID, err)
		}
		if err := validateRuntimeStatus(response.DeploymentStatus, plan, stage, "active", true); err != nil {
			return fmt.Errorf("node %q: %w", stage.NodeID, err)
		}
	}
	return nil
}

func (c *DeploymentController) drainPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	identity := runtimeDeploymentIdentity(plan)
	var failures []string
	for _, stage := range plan.Stages() {
		response, err := c.runtimeClient.Drain(ctx, stage.APIURL, identity)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q: %v", stage.NodeID, err))
			continue
		}
		if !runtimeDeploymentIdentityMatches(response.Deployment, identity) {
			failures = append(failures, fmt.Sprintf("node %q acknowledged a different deployment identity", stage.NodeID))
		}
	}
	return joinedFailures(failures)
}

func (c *DeploymentController) unloadPlan(ctx context.Context, plan clusterplan.DeploymentPlan) error {
	identity := runtimeDeploymentIdentity(plan)
	var failures []string
	for _, stage := range plan.Stages() {
		response, err := c.runtimeClient.Unload(ctx, stage.APIURL, identity)
		if err != nil {
			failures = append(failures, fmt.Sprintf("node %q: %v", stage.NodeID, err))
			continue
		}
		if !runtimeDeploymentIdentityMatches(response.Deployment, identity) {
			failures = append(failures, fmt.Sprintf("node %q acknowledged a different deployment identity", stage.NodeID))
			continue
		}
		if response.Resident || response.Active || response.State != "idle" {
			failures = append(failures, fmt.Sprintf("node %q did not release the deployment", stage.NodeID))
		}
	}
	return joinedFailures(failures)
}

func joinedFailures(failures []string) error {
	if len(failures) == 0 {
		return nil
	}
	sort.Strings(failures)
	return errors.New(strings.Join(failures, "; "))
}

func plansEquivalent(left, right clusterplan.DeploymentPlan) bool {
	return left.Model() == right.Model() && slices.Equal(left.Stages(), right.Stages())
}

func activePlanHealthy(
	plan clusterplan.DeploymentPlan,
	members []membership.Member,
	now time.Time,
	staleAfter time.Duration,
) bool {
	fresh := make(map[string]membership.Member, len(members))
	for _, member := range members {
		member = membership.Normalize(member)
		if member.Valid() && !member.IsStale(now, staleAfter) {
			fresh[member.NodeID] = member
		}
	}
	for _, stage := range plan.Stages() {
		member, ok := fresh[stage.NodeID]
		if !ok || member.APIURL != stage.APIURL {
			return false
		}
	}
	return true
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
	wantIdentity := runtimeDeploymentIdentity(plan)
	if !runtimeDeploymentIdentityMatches(status.Deployment, wantIdentity) {
		return fmt.Errorf(
			"runtime reports deployment %q epoch %d model %q sha256 %q, want deployment %q epoch %d model %q sha256 %q for stage %d",
			status.Deployment.DeploymentID, status.Deployment.Epoch,
			status.Deployment.ModelID, status.Deployment.ModelSHA256,
			wantIdentity.DeploymentID, wantIdentity.Epoch,
			wantIdentity.ModelID, wantIdentity.ModelSHA256, stage.StageIndex,
		)
	}
	if err := validateRuntimeModelMemory(status, plan, stage, active); err != nil {
		return fmt.Errorf("stage %d model residency: %w", stage.StageIndex, err)
	}
	return nil
}

func runtimeDeploymentIdentity(plan clusterplan.DeploymentPlan) runtimebridge.DeploymentIdentity {
	return runtimebridge.DeploymentIdentity{
		DeploymentID: plan.Identity().DeploymentID,
		Epoch:        plan.Identity().Epoch,
		ModelID:      plan.Model().ModelID,
		ModelSHA256:  plan.Model().ModelSHA256,
	}
}

func runtimeDeploymentIdentityMatches(actual *runtimebridge.DeploymentIdentity, expected runtimebridge.DeploymentIdentity) bool {
	return actual != nil && *actual == expected
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
