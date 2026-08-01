package clusterplan

const (
	ReasonInvalidLayerCount            Reason = "invalid_layer_count"
	ReasonInvalidStageCount            Reason = "invalid_stage_count"
	ReasonInvalidStageLayerCounts      Reason = "invalid_stage_layer_counts"
	ReasonStageCountExceedsLayers      Reason = "stage_count_exceeds_layers"
	ReasonInvalidStageIndices          Reason = "invalid_stage_indices"
	ReasonInvalidLayerRanges           Reason = "invalid_layer_ranges"
	ReasonIncompleteRuntimeAssignments Reason = "incomplete_runtime_assignments"
)

func validatePlanningRequest(layerCount int, requestedStageCount int) Reason {
	if layerCount <= 0 {
		return ReasonInvalidLayerCount
	}
	if requestedStageCount < 0 {
		return ReasonInvalidStageCount
	}
	if requestedStageCount > layerCount {
		return ReasonStageCountExceedsLayers
	}
	return ""
}

func policyStageCount(policy Policy) int {
	if policy.StageCount != 0 {
		return policy.StageCount
	}
	if len(policy.StageLayerCounts) > 0 {
		return len(policy.StageLayerCounts)
	}
	return 0
}

func validateStageLayerCounts(layerCount int, stageCount int, counts []int) Reason {
	if len(counts) == 0 {
		return ""
	}
	if stageCount != len(counts) {
		return ReasonInvalidStageLayerCounts
	}
	total := 0
	for _, count := range counts {
		if count <= 0 {
			return ReasonInvalidStageLayerCounts
		}
		total += count
	}
	if total != layerCount {
		return ReasonInvalidStageLayerCounts
	}
	return ""
}

// validateStagePlan enforces the global invariants that individual runtime
// stages cannot verify from their local assignment alone.
func validateStagePlan(stages []Stage, layerCount int) Reason {
	if layerCount <= 0 {
		return ReasonInvalidLayerCount
	}
	if len(stages) == 0 {
		return ReasonInvalidStageCount
	}
	if len(stages) > layerCount {
		return ReasonStageCountExceedsLayers
	}

	expectedStart := 0
	for index, stage := range stages {
		if stage.StageIndex != index || stage.StageCount != len(stages) {
			return ReasonInvalidStageIndices
		}
		if stage.LayerStart != expectedStart || stage.LayerEnd <= stage.LayerStart || stage.LayerEnd > layerCount {
			return ReasonInvalidLayerRanges
		}
		expectedStart = stage.LayerEnd
	}
	if expectedStart != layerCount {
		return ReasonInvalidLayerRanges
	}
	return ""
}

func verifiedFullModelReplica(candidates []Placement, layerCount int) (Placement, bool) {
	for _, candidate := range candidates {
		if candidate.RuntimeStageAssigned &&
			candidate.RuntimeStageIndex == 0 &&
			candidate.RuntimeStageCount == 1 &&
			candidate.RuntimeLayerStart == 0 &&
			candidate.RuntimeLayerEnd == layerCount {
			return candidate, true
		}
	}
	return Placement{}, false
}
