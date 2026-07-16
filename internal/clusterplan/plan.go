package clusterplan

import (
	"fmt"
	"net"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

type Topology string

const (
	TopologyColocated   Topology = "colocated"
	TopologyDistributed Topology = "distributed"
)

type Reason string

const (
	ReasonCandidate                   Reason = "candidate"
	ReasonNoEligibleMembers           Reason = "no_eligible_members"
	ReasonInsufficientMemory          Reason = "insufficient_memory"
	ReasonMissingCompute              Reason = "missing_compute"
	ReasonInsufficientStages          Reason = "insufficient_stages"
	ReasonColocatedStagesDisallowed   Reason = "colocated_stages_disallowed"
	ReasonPipelineParallelUnsupported Reason = "pipeline_parallel_unsupported"
)

const (
	defaultPipelineStageCount = 2
	warningColocated          = "colocated stages validate orchestration, not distributed physical compute"
	warningUsingFullReplica   = "no complete explicit pipeline assignment found; using one full-model replica"
)

type Policy struct {
	AllowColocatedStages bool `json:"allow_colocated_stages"`
	StageCount           int  `json:"stage_count,omitempty"`
}

type Request struct {
	Model      cluster.ModelProfile
	Members    []membership.Member
	Now        time.Time
	StaleAfter time.Duration
	Policy     Policy
}

type RoutePreview struct {
	Model             string                `json:"model"`
	Valid             bool                  `json:"valid"`
	Reason            Reason                `json:"reason,omitempty"`
	Mode              cluster.ExecutionMode `json:"mode,omitempty"`
	Topology          Topology              `json:"topology,omitempty"`
	StageCount        int                   `json:"stage_count,omitempty"`
	LogicalNodeCount  int                   `json:"logical_node_count,omitempty"`
	PhysicalHostCount int                   `json:"physical_host_count,omitempty"`
	Placements        []Placement           `json:"placements,omitempty"`
	Stages            []Stage               `json:"stages,omitempty"`
	Warnings          []string              `json:"warnings,omitempty"`
}

type Placement struct {
	NodeID         string `json:"node_id"`
	NodeName       string `json:"node_name"`
	Hostname       string `json:"hostname,omitempty"`
	PhysicalHostID string `json:"physical_host_id"`
	APIURL         string `json:"api_url"`
	Valid          bool   `json:"valid"`
	MemoryOK       bool   `json:"memory_ok"`
	ComputeOK      bool   `json:"compute_ok"`
	Reason         Reason `json:"reason"`

	RuntimeStageAssigned bool `json:"runtime_stage_assigned,omitempty"`
	RuntimeStageIndex    int  `json:"runtime_stage_index,omitempty"`
	RuntimeStageCount    int  `json:"runtime_stage_count,omitempty"`
	RuntimeLayerStart    int  `json:"runtime_layer_start,omitempty"`
	RuntimeLayerEnd      int  `json:"runtime_layer_end,omitempty"`
}

type Stage struct {
	StageIndex     int    `json:"stage_index"`
	StageCount     int    `json:"stage_count"`
	NodeID         string `json:"node_id"`
	NodeName       string `json:"node_name"`
	Hostname       string `json:"hostname,omitempty"`
	PhysicalHostID string `json:"physical_host_id"`
	APIURL         string `json:"api_url"`
	LayerStart     int    `json:"layer_start"`
	LayerEnd       int    `json:"layer_end"`
}

type LayerRange struct {
	Start int
	End   int
}

func Preview(req Request) RoutePreview {
	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}

	placements, candidates := candidatePlacements(req.Model, req.Members, now, req.StaleAfter)
	preview := RoutePreview{Model: req.Model.ID, Placements: placements}
	if reason := validatePlanningRequest(req.Model.LayerCount, req.Policy.StageCount); reason != "" {
		preview.Reason = reason
		return preview
	}
	if len(candidates) == 0 {
		preview.Reason = ReasonNoEligibleMembers
		return preview
	}

	pipelineSupported := supportsMode(req.Model, cluster.ExecutionModePipelineParallel)
	if pipelineSupported {
		if explicit, ok := explicitPipelinePlacements(candidates, req.Model.LayerCount, req.Policy.StageCount); ok {
			return pipelinePreview(preview, explicit, req.Policy, req.Model.LayerCount)
		}
	}

	if hasRuntimeAssignments(candidates) {
		if replica, ok := verifiedFullModelReplica(candidates, req.Model.LayerCount); ok {
			preview.Warnings = append(preview.Warnings, warningUsingFullReplica)
			return dataParallelPreview(preview, replica, req.Model.LayerCount)
		}
		preview.Reason = ReasonIncompleteRuntimeAssignments
		return preview
	}

	requestedStageCount := req.Policy.StageCount
	if requestedStageCount == 0 {
		requestedStageCount = defaultPipelineStageCount
		if requestedStageCount > req.Model.LayerCount {
			requestedStageCount = req.Model.LayerCount
		}
	}
	if !pipelineSupported || requestedStageCount <= 1 || len(candidates) == 1 {
		if !pipelineSupported && requestedStageCount > 1 {
			preview.Warnings = append(preview.Warnings, string(ReasonPipelineParallelUnsupported))
		}
		return dataParallelPreview(preview, candidates[0], req.Model.LayerCount)
	}
	if requestedStageCount > len(candidates) {
		preview.Reason = ReasonInsufficientStages
		return preview
	}

	distributed := selectDistinctPhysicalHosts(candidates, requestedStageCount)
	if len(distributed) == requestedStageCount {
		return pipelinePreview(preview, distributed, req.Policy, req.Model.LayerCount)
	}
	return pipelinePreview(preview, candidates[:requestedStageCount], req.Policy, req.Model.LayerCount)
}

func dataParallelPreview(preview RoutePreview, placement Placement, layerCount int) RoutePreview {
	preview.Mode = cluster.ExecutionModeDataParallel
	preview.Stages = buildStages([]Placement{placement}, layerCount)
	if reason := validateStagePlan(preview.Stages, layerCount); reason != "" {
		preview.Reason = reason
		return finalizeTopology(preview)
	}
	preview.Valid = true
	return finalizeTopology(preview)
}

func pipelinePreview(preview RoutePreview, placements []Placement, policy Policy, layerCount int) RoutePreview {
	preview.Mode = cluster.ExecutionModePipelineParallel
	preview.Stages = buildStages(placements, layerCount)
	if reason := validateStagePlan(preview.Stages, layerCount); reason != "" {
		preview.Reason = reason
		return finalizeTopology(preview)
	}
	if topologyForPlacements(placements) == TopologyColocated && len(placements) > 1 && !policy.AllowColocatedStages {
		preview.Reason = ReasonColocatedStagesDisallowed
		preview.Warnings = append(preview.Warnings, warningColocated)
		return finalizeTopology(preview)
	}

	preview.Valid = true
	if topologyForPlacements(placements) == TopologyColocated && len(placements) > 1 {
		preview.Warnings = append(preview.Warnings, warningColocated)
	}
	return finalizeTopology(preview)
}

func finalizeTopology(preview RoutePreview) RoutePreview {
	preview.StageCount = len(preview.Stages)
	logicalNodes := map[string]bool{}
	physicalHosts := map[string]bool{}
	for _, stage := range preview.Stages {
		logicalNodes[stage.NodeID] = true
		physicalHosts[stage.PhysicalHostID] = true
	}
	preview.LogicalNodeCount = len(logicalNodes)
	preview.PhysicalHostCount = len(physicalHosts)
	preview.Topology = topologyForStages(preview.Stages)
	return preview
}

func topologyForStages(stages []Stage) Topology {
	seen := map[string]bool{}
	for _, stage := range stages {
		if seen[stage.PhysicalHostID] {
			return TopologyColocated
		}
		seen[stage.PhysicalHostID] = true
	}
	if len(seen) <= 1 {
		return TopologyColocated
	}
	return TopologyDistributed
}

func topologyForPlacements(placements []Placement) Topology {
	seen := map[string]bool{}
	for _, placement := range placements {
		if seen[placement.PhysicalHostID] {
			return TopologyColocated
		}
		seen[placement.PhysicalHostID] = true
	}
	if len(seen) <= 1 {
		return TopologyColocated
	}
	return TopologyDistributed
}

func candidatePlacements(model cluster.ModelProfile, members []membership.Member, now time.Time, staleAfter time.Duration) ([]Placement, []Placement) {
	placements := make([]Placement, 0, len(members))
	candidates := []Placement{}
	for _, member := range members {
		member = membership.Normalize(member)
		if !member.Valid() || member.IsStale(now, staleAfter) {
			continue
		}
		placement := placementForMember(model, member)
		placements = append(placements, placement)
		if placement.Valid {
			candidates = append(candidates, placement)
		}
	}
	sortPlacements(placements)
	sortPlacements(candidates)
	return placements, candidates
}

func placementForMember(model cluster.ModelProfile, member membership.Member) Placement {
	memoryOK := memoryOK(model, member)
	computeOK := computeOK(model, member)
	placement := Placement{
		NodeID:         member.NodeID,
		NodeName:       member.NodeName,
		Hostname:       member.Hostname,
		PhysicalHostID: PhysicalHostID(member),
		APIURL:         member.APIURL,
		Valid:          memoryOK && computeOK,
		MemoryOK:       memoryOK,
		ComputeOK:      computeOK,
		Reason:         placementReason(memoryOK, computeOK, model.PreferredCompute),
	}
	if assignment, ok := runtimeAssignment(member); ok {
		placement.RuntimeStageAssigned = true
		placement.RuntimeStageIndex = assignment.StageIndex
		placement.RuntimeStageCount = assignment.StageCount
		placement.RuntimeLayerStart = assignment.LayerStart
		placement.RuntimeLayerEnd = assignment.LayerEnd
	}
	return placement
}

type runtimeStageAssignment struct {
	StageIndex int
	StageCount int
	LayerStart int
	LayerEnd   int
}

func runtimeAssignment(member membership.Member) (runtimeStageAssignment, bool) {
	stageIndex, ok := intCapability(member.Capabilities, cluster.CapabilityRuntimeStageIndex)
	if !ok {
		return runtimeStageAssignment{}, false
	}
	stageCount, ok := intCapability(member.Capabilities, cluster.CapabilityRuntimeStageCount)
	if !ok {
		return runtimeStageAssignment{}, false
	}
	layerStart, ok := intCapability(member.Capabilities, cluster.CapabilityRuntimeLayerStart)
	if !ok {
		return runtimeStageAssignment{}, false
	}
	layerEnd, ok := intCapability(member.Capabilities, cluster.CapabilityRuntimeLayerEnd)
	if !ok {
		return runtimeStageAssignment{}, false
	}
	if stageCount <= 0 || stageIndex < 0 || stageIndex >= stageCount || layerEnd <= layerStart {
		return runtimeStageAssignment{}, false
	}
	return runtimeStageAssignment{StageIndex: stageIndex, StageCount: stageCount, LayerStart: layerStart, LayerEnd: layerEnd}, true
}

func hasRuntimeAssignments(candidates []Placement) bool {
	for _, candidate := range candidates {
		if candidate.RuntimeStageAssigned {
			return true
		}
	}
	return false
}

func explicitPipelinePlacements(candidates []Placement, layerCount int, requestedStageCount int) ([]Placement, bool) {
	byStageCount := map[int][]Placement{}
	for _, candidate := range candidates {
		if !candidate.RuntimeStageAssigned || candidate.RuntimeStageCount < 2 {
			continue
		}
		if requestedStageCount > 0 && candidate.RuntimeStageCount != requestedStageCount {
			continue
		}
		byStageCount[candidate.RuntimeStageCount] = append(byStageCount[candidate.RuntimeStageCount], candidate)
	}

	stageCounts := make([]int, 0, len(byStageCount))
	for stageCount := range byStageCount {
		stageCounts = append(stageCounts, stageCount)
	}
	sort.Ints(stageCounts)

	for _, stageCount := range stageCounts {
		placements := byStageCount[stageCount]
		if len(placements) < stageCount {
			continue
		}
		ordered := make([]Placement, stageCount)
		seen := make([]bool, stageCount)
		for _, placement := range placements {
			idx := placement.RuntimeStageIndex
			if idx < 0 || idx >= stageCount || seen[idx] {
				continue
			}
			ordered[idx] = placement
			seen[idx] = true
		}
		complete := true
		for _, present := range seen {
			if !present {
				complete = false
				break
			}
		}
		if complete && runtimeRangesCoverModel(ordered, layerCount) {
			return ordered, true
		}
	}
	return nil, false
}

func runtimeRangesCoverModel(placements []Placement, layerCount int) bool {
	expectedStart := 0
	for _, placement := range placements {
		if placement.RuntimeLayerStart != expectedStart || placement.RuntimeLayerEnd <= placement.RuntimeLayerStart {
			return false
		}
		expectedStart = placement.RuntimeLayerEnd
	}
	return layerCount <= 0 || expectedStart == layerCount
}

func memoryOK(model cluster.ModelProfile, member membership.Member) bool {
	if model.MinMemoryGB <= 0 {
		return true
	}
	return floatCapability(member.Capabilities, cluster.CapabilityMemoryGB) >= model.MinMemoryGB
}

func computeOK(model cluster.ModelProfile, member membership.Member) bool {
	if model.PreferredCompute == nil || *model.PreferredCompute == "" {
		return true
	}
	return containsStringCapability(member.Capabilities, cluster.CapabilityComputeBackends, string(*model.PreferredCompute))
}

func placementReason(memoryOK bool, computeOK bool, compute *cluster.ComputeBackend) Reason {
	switch {
	case !memoryOK:
		return ReasonInsufficientMemory
	case !computeOK && compute != nil:
		return Reason(fmt.Sprintf("%s:%s", ReasonMissingCompute, *compute))
	case !computeOK:
		return ReasonMissingCompute
	default:
		return ReasonCandidate
	}
}

func sortPlacements(placements []Placement) {
	sort.SliceStable(placements, func(i, j int) bool {
		left := placements[i]
		right := placements[j]
		if left.RuntimeStageAssigned && right.RuntimeStageAssigned && left.RuntimeStageCount == right.RuntimeStageCount && left.RuntimeStageIndex != right.RuntimeStageIndex {
			return left.RuntimeStageIndex < right.RuntimeStageIndex
		}
		if left.PhysicalHostID != right.PhysicalHostID {
			return left.PhysicalHostID < right.PhysicalHostID
		}
		if left.NodeID != right.NodeID {
			return left.NodeID < right.NodeID
		}
		return left.NodeName < right.NodeName
	})
}

func selectDistinctPhysicalHosts(candidates []Placement, count int) []Placement {
	selected := make([]Placement, 0, count)
	seen := map[string]bool{}
	for _, candidate := range candidates {
		if seen[candidate.PhysicalHostID] {
			continue
		}
		seen[candidate.PhysicalHostID] = true
		selected = append(selected, candidate)
		if len(selected) == count {
			return selected
		}
	}
	return selected
}

func buildStages(placements []Placement, layerCount int) []Stage {
	ranges := AssignLayerRanges(layerCount, len(placements))
	if len(ranges) != len(placements) {
		return nil
	}
	stages := make([]Stage, 0, len(placements))
	for i, placement := range placements {
		stageIndex := i
		stageCount := len(placements)
		layerStart := ranges[i].Start
		layerEnd := ranges[i].End
		if placement.RuntimeStageAssigned {
			stageIndex = placement.RuntimeStageIndex
			stageCount = placement.RuntimeStageCount
			layerStart = placement.RuntimeLayerStart
			layerEnd = placement.RuntimeLayerEnd
		}
		stages = append(stages, Stage{
			StageIndex:     stageIndex,
			StageCount:     stageCount,
			NodeID:         placement.NodeID,
			NodeName:       placement.NodeName,
			Hostname:       placement.Hostname,
			PhysicalHostID: placement.PhysicalHostID,
			APIURL:         placement.APIURL,
			LayerStart:     layerStart,
			LayerEnd:       layerEnd,
		})
	}
	return stages
}

func AssignLayerRanges(layerCount int, stageCount int) []LayerRange {
	if layerCount <= 0 || stageCount <= 0 || stageCount > layerCount {
		return nil
	}

	ranges := make([]LayerRange, 0, stageCount)
	base := layerCount / stageCount
	remainder := layerCount % stageCount
	start := 0
	for i := 0; i < stageCount; i++ {
		width := base
		if i < remainder {
			width++
		}
		end := start + width
		ranges = append(ranges, LayerRange{Start: start, End: end})
		start = end
	}
	return ranges
}

func supportsMode(model cluster.ModelProfile, mode cluster.ExecutionMode) bool {
	for _, candidate := range model.PlacementModes {
		if candidate == mode {
			return true
		}
	}
	return false
}

func PhysicalHostID(member membership.Member) string {
	if host := strings.TrimSpace(member.Hostname); host != "" {
		return host
	}
	if host := hostFromURL(member.APIURL); host != "" {
		return host
	}
	if name := strings.TrimSpace(member.NodeName); name != "" {
		return name
	}
	return member.NodeID
}

func hostFromURL(rawURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Host == "" {
		return ""
	}
	host := parsed.Hostname()
	if host != "" {
		return host
	}
	host, _, err = net.SplitHostPort(parsed.Host)
	if err == nil {
		return host
	}
	return parsed.Host
}

func floatCapability(caps map[string]any, key string) float64 {
	value, ok := caps[key]
	if !ok {
		return 0
	}
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case uint64:
		return float64(typed)
	default:
		return 0
	}
}

func intCapability(caps map[string]any, key string) (int, bool) {
	value, ok := caps[key]
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case int:
		return typed, true
	case int64:
		return int(typed), true
	case int32:
		return int(typed), true
	case float64:
		return int(typed), true
	case float32:
		return int(typed), true
	case uint64:
		return int(typed), true
	default:
		return 0, false
	}
}

func containsStringCapability(caps map[string]any, key string, expected string) bool {
	value, ok := caps[key]
	if !ok {
		return false
	}
	switch typed := value.(type) {
	case []string:
		return containsString(typed, expected)
	case []any:
		for _, item := range typed {
			if text, ok := item.(string); ok && text == expected {
				return true
			}
		}
	case []cluster.ComputeBackend:
		for _, item := range typed {
			if string(item) == expected {
				return true
			}
		}
	}
	return false
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}
