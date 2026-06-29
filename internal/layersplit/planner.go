package layersplit

import (
	"fmt"
	"math"
	"sort"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

type StageRole string

const (
	StageRoleFirst  StageRole = "first"
	StageRoleMiddle StageRole = "middle"
	StageRoleLast   StageRole = "last"
)

type NodeCandidate struct {
	NodeName    string              `json:"node_name"`
	BackendID   string              `json:"backend_id,omitempty"`
	BackendKind cluster.RuntimeKind `json:"backend_kind,omitempty"`
	BaseURL     string              `json:"-"`
	Weight      float64             `json:"weight"`
}

type Plan struct {
	ModelID    string            `json:"model_id"`
	Mode       cluster.RouteMode `json:"mode"`
	LayerCount int               `json:"layer_count"`
	Stages     []Stage           `json:"stages"`
}

type Stage struct {
	Index       int                 `json:"index"`
	NodeName    string              `json:"node_name"`
	BackendID   string              `json:"backend_id,omitempty"`
	BackendKind cluster.RuntimeKind `json:"backend_kind,omitempty"`
	BaseURL     string              `json:"-"`
	Role        StageRole           `json:"role"`
	LayerStart  int                 `json:"layer_start"`
	LayerEnd    int                 `json:"layer_end"`
	LayerCount  int                 `json:"layer_count"`
	Weight      float64             `json:"weight"`
}

func PlanForModel(model cluster.ModelProfile, candidates []NodeCandidate) (Plan, error) {
	return PlanForCandidates(model.ID, model.LayerCount, candidates)
}

func PlanForCandidates(modelID string, layerCount int, candidates []NodeCandidate) (Plan, error) {
	if modelID == "" {
		return Plan{}, fmt.Errorf("model id is required")
	}
	if layerCount < 2 {
		return Plan{}, fmt.Errorf("layer split requires at least two layers")
	}
	candidates = normalizeCandidates(candidates)
	if len(candidates) < 2 {
		return Plan{}, fmt.Errorf("layer split requires at least two candidate nodes")
	}

	stageCount := minInt(len(candidates), layerCount)
	candidates = candidates[:stageCount]
	counts := allocateLayers(layerCount, candidates)

	stages := make([]Stage, 0, stageCount)
	start := 0
	for index, candidate := range candidates {
		count := counts[index]
		end := start + count
		stages = append(stages, Stage{
			Index:       index,
			NodeName:    candidate.NodeName,
			BackendID:   candidate.BackendID,
			BackendKind: candidate.BackendKind,
			BaseURL:     candidate.BaseURL,
			Role:        stageRole(index, stageCount),
			LayerStart:  start,
			LayerEnd:    end,
			LayerCount:  count,
			Weight:      candidate.Weight,
		})
		start = end
	}

	return Plan{
		ModelID:    modelID,
		Mode:       cluster.RouteModeLayerSplit,
		LayerCount: layerCount,
		Stages:     stages,
	}, nil
}

func normalizeCandidates(candidates []NodeCandidate) []NodeCandidate {
	normalized := make([]NodeCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.NodeName == "" {
			continue
		}
		if !ValidWeight(candidate.Weight) {
			candidate.Weight = 1
		}
		normalized = append(normalized, candidate)
	}
	return normalized
}

func allocateLayers(layerCount int, candidates []NodeCandidate) []int {
	counts := make([]int, len(candidates))
	for index := range counts {
		counts[index] = 1
	}

	remaining := layerCount - len(candidates)
	if remaining == 0 {
		return counts
	}

	totalWeight := 0.0
	for _, candidate := range candidates {
		totalWeight += candidate.Weight
	}

	type remainder struct {
		index int
		value float64
	}
	remainders := make([]remainder, 0, len(candidates))
	allocated := 0
	for index, candidate := range candidates {
		ideal := (candidate.Weight / totalWeight) * float64(remaining)
		whole := int(math.Floor(ideal))
		counts[index] += whole
		allocated += whole
		remainders = append(remainders, remainder{index: index, value: ideal - float64(whole)})
	}

	sort.SliceStable(remainders, func(left int, right int) bool {
		if remainders[left].value == remainders[right].value {
			return remainders[left].index < remainders[right].index
		}
		return remainders[left].value > remainders[right].value
	})

	for index := 0; index < remaining-allocated; index++ {
		counts[remainders[index].index]++
	}
	return counts
}

func stageRole(index int, stageCount int) StageRole {
	switch {
	case index == 0:
		return StageRoleFirst
	case index == stageCount-1:
		return StageRoleLast
	default:
		return StageRoleMiddle
	}
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}
