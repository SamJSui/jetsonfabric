package clusterplan

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

// DeploymentCompatibility records the runtime facts that were required to
// agree before a deployment plan was accepted.
type DeploymentCompatibility struct {
	Architecture     string                 `json:"architecture"`
	RuntimeRevision  string                 `json:"runtime_revision"`
	LlamaCPPRevision string                 `json:"llama_cpp_revision,omitempty"`
	ComputeBackend   cluster.ComputeBackend `json:"compute_backend,omitempty"`
	CUDAActive       bool                   `json:"cuda_active"`
}

type DeploymentBuildRequest struct {
	Identity   DeploymentIdentity
	Model      cluster.ModelProfile
	Members    []membership.Member
	Now        time.Time
	StaleAfter time.Duration
	Policy     Policy
}

type DeploymentBuildResult struct {
	Plan          DeploymentPlan          `json:"plan"`
	Preview       RoutePreview            `json:"preview"`
	Compatibility DeploymentCompatibility `json:"compatibility"`
}

// BuildDeploymentPlan constructs an immutable epoch from one fresh membership
// snapshot. Membership changes after this call do not mutate the returned plan.
func BuildDeploymentPlan(req DeploymentBuildRequest) (DeploymentBuildResult, error) {
	model := req.Model
	model.ID = strings.TrimSpace(model.ID)
	model.ArtifactPath = strings.TrimSpace(model.ArtifactPath)
	model.ArtifactSHA256 = strings.ToLower(strings.TrimSpace(model.ArtifactSHA256))
	if model.ID == "" {
		return DeploymentBuildResult{}, fmt.Errorf("model_id is required")
	}
	if !validSHA256(model.ArtifactSHA256) {
		return DeploymentBuildResult{}, fmt.Errorf("model %q requires an exact artifact_sha256", model.ID)
	}
	if model.LayerCount <= 0 {
		return DeploymentBuildResult{}, fmt.Errorf("model %q requires a positive layer_count", model.ID)
	}
	if !supportsMode(model, cluster.ExecutionModePipelineParallel) {
		return DeploymentBuildResult{}, fmt.Errorf("model %q does not support pipeline_parallel", model.ID)
	}

	engine, err := selectDeploymentEngine(model)
	if err != nil {
		return DeploymentBuildResult{}, err
	}
	if engine == cluster.EngineLlamaCPP && model.ArtifactPath == "" {
		return DeploymentBuildResult{}, fmt.Errorf("model %q requires artifact_path for llama.cpp deployment", model.ID)
	}

	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	members, compatibility, err := compatibleDeploymentMembers(
		model,
		engine,
		req.Members,
		now,
		req.StaleAfter,
		requiredDeploymentStages(req.Policy),
	)
	if err != nil {
		return DeploymentBuildResult{}, err
	}

	preview := PreviewPipeline(Request{
		Model:      model,
		Members:    members,
		Now:        now,
		StaleAfter: req.StaleAfter,
		Policy:     req.Policy,
	})
	if !preview.Valid {
		return DeploymentBuildResult{}, fmt.Errorf("no valid deployment route for model %q: %s", model.ID, preview.Reason)
	}
	if preview.Mode != cluster.ExecutionModePipelineParallel || preview.StageCount < 1 {
		return DeploymentBuildResult{}, fmt.Errorf("deployment route must use pipeline_parallel with at least one stage")
	}

	plan, err := NewDeploymentPlan(DeploymentPlanSpec{
		Identity: req.Identity,
		Model: DeploymentModelIdentity{
			ModelID:       model.ID,
			ModelSHA256:   model.ArtifactSHA256,
			Engine:        engine,
			ExecutionMode: cluster.ExecutionModePipelineParallel,
			LayerCount:    model.LayerCount,
		},
		Stages: preview.Stages,
	})
	if err != nil {
		return DeploymentBuildResult{}, err
	}
	return DeploymentBuildResult{Plan: plan, Preview: preview, Compatibility: compatibility}, nil
}

func selectDeploymentEngine(model cluster.ModelProfile) (cluster.Engine, error) {
	engines := make([]cluster.Engine, 0, len(model.SupportedEngines))
	for _, engine := range model.SupportedEngines {
		engine = cluster.Engine(strings.TrimSpace(string(engine)))
		if engine != "" {
			engines = append(engines, engine)
		}
	}
	if len(engines) == 0 {
		return "", fmt.Errorf("model %q has no supported deployment engine", model.ID)
	}
	sort.SliceStable(engines, func(i, j int) bool { return engines[i] < engines[j] })
	return engines[0], nil
}

func requiredDeploymentStages(policy Policy) int {
	if policy.StageCount > 0 {
		return policy.StageCount
	}
	return 1
}

type compatibilityGroup struct {
	members       []membership.Member
	compatibility DeploymentCompatibility
}

func compatibleDeploymentMembers(
	model cluster.ModelProfile,
	engine cluster.Engine,
	members []membership.Member,
	now time.Time,
	staleAfter time.Duration,
	requiredStages int,
) ([]membership.Member, DeploymentCompatibility, error) {
	groups := map[string]*compatibilityGroup{}
	for _, member := range members {
		member = membership.Normalize(member)
		if !member.Valid() || member.IsStale(now, staleAfter) {
			continue
		}
		compatibility, ok := memberDeploymentCompatibility(model, engine, member)
		if !ok {
			continue
		}
		key := compatibilityKey(compatibility, model.PreferredCompute != nil)
		group := groups[key]
		if group == nil {
			group = &compatibilityGroup{compatibility: compatibility}
			groups[key] = group
		}
		group.members = append(group.members, member)
	}

	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var selected *compatibilityGroup
	for _, key := range keys {
		group := groups[key]
		if len(group.members) < requiredStages {
			continue
		}
		if selected == nil || len(group.members) > len(selected.members) {
			selected = group
		}
	}
	if selected == nil {
		return nil, DeploymentCompatibility{}, fmt.Errorf(
			"need %d fresh runtimes with matching architecture, runtime revision, engine revision, execution mode, and compute compatibility",
			requiredStages,
		)
	}
	return selected.members, selected.compatibility, nil
}

func memberDeploymentCompatibility(
	model cluster.ModelProfile,
	engine cluster.Engine,
	member membership.Member,
) (DeploymentCompatibility, bool) {
	architecture := strings.TrimSpace(member.Arch)
	runtimeRevision := capabilityText(member.Capabilities, cluster.CapabilityRuntimeRevision)
	llamaRevision := capabilityText(member.Capabilities, cluster.CapabilityRuntimeLlamaCPPRevision)
	backend := cluster.ComputeBackend(capabilityText(member.Capabilities, cluster.CapabilityRuntimeComputeBackend))
	if architecture == "" || runtimeRevision == "" {
		return DeploymentCompatibility{}, false
	}
	if cluster.Engine(capabilityText(member.Capabilities, cluster.CapabilityRuntimeEngine)) != engine {
		return DeploymentCompatibility{}, false
	}
	if cluster.ExecutionMode(capabilityText(member.Capabilities, cluster.CapabilityRuntimeExecutionMode)) != cluster.ExecutionModePipelineParallel {
		return DeploymentCompatibility{}, false
	}
	if engine == cluster.EngineLlamaCPP && llamaRevision == "" {
		return DeploymentCompatibility{}, false
	}
	if backend != cluster.ComputeBackendCPU && backend != cluster.ComputeBackendCUDA {
		return DeploymentCompatibility{}, false
	}

	cudaActive := capabilityBool(member.Capabilities, cluster.CapabilityRuntimeCUDAActive)
	if model.PreferredCompute != nil && *model.PreferredCompute != "" {
		if backend != *model.PreferredCompute {
			return DeploymentCompatibility{}, false
		}
		if backend == cluster.ComputeBackendCUDA && !cudaActive {
			return DeploymentCompatibility{}, false
		}
	}
	return DeploymentCompatibility{
		Architecture:     architecture,
		RuntimeRevision:  runtimeRevision,
		LlamaCPPRevision: llamaRevision,
		ComputeBackend:   backend,
		CUDAActive:       cudaActive,
	}, true
}

func compatibilityKey(value DeploymentCompatibility, includeBackend bool) string {
	backend := "mixed"
	cuda := "mixed"
	if includeBackend {
		backend = string(value.ComputeBackend)
		cuda = fmt.Sprintf("%t", value.CUDAActive)
	}
	return strings.Join([]string{
		value.Architecture,
		value.RuntimeRevision,
		value.LlamaCPPRevision,
		backend,
		cuda,
	}, "|")
}

func capabilityText(capabilities map[string]any, key string) string {
	if capabilities == nil {
		return ""
	}
	value, ok := capabilities[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func capabilityBool(capabilities map[string]any, key string) bool {
	if capabilities == nil {
		return false
	}
	value, ok := capabilities[key]
	if !ok {
		return false
	}
	result, ok := value.(bool)
	return ok && result
}
