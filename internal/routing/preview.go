package routing

import (
	"fmt"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

type Reason string

const (
	ReasonUnknownModel       Reason = "unknown_model"
	ReasonInsufficientMemory Reason = "insufficient_memory"
	ReasonMissingCompute     Reason = "missing_compute"
	ReasonCandidate          Reason = "candidate"
)

type PlacementPreview struct {
	NodeName  string `json:"node_name"`
	Valid     bool   `json:"valid"`
	MemoryOK  bool   `json:"memory_ok"`
	ComputeOK bool   `json:"compute_ok"`
	Reason    Reason `json:"reason"`
}

type RoutePreview struct {
	Model      string             `json:"model"`
	Valid      bool               `json:"valid"`
	Reason     Reason             `json:"reason,omitempty"`
	Placements []PlacementPreview `json:"placements,omitempty"`
}

func Preview(model cluster.ModelProfile, nodes []cluster.NodeRecord) RoutePreview {
	placements := make([]PlacementPreview, 0, len(nodes))
	for _, node := range nodes {
		memory := floatCapability(node.Capabilities, cluster.CapabilityMemoryGB)

		computeOK := true
		if model.PreferredCompute != nil && *model.PreferredCompute != "" {
			computeOK = containsStringCapability(
				node.Capabilities,
				cluster.CapabilityComputeBackends,
				string(*model.PreferredCompute),
			)
		}

		memoryOK := memory >= model.MinMemoryGB

		placements = append(placements, PlacementPreview{
			NodeName:  node.NodeName,
			Valid:     memoryOK && computeOK,
			MemoryOK:  memoryOK,
			ComputeOK: computeOK,
			Reason:    routeReason(memoryOK, computeOK, model.PreferredCompute),
		})
	}
	return RoutePreview{Model: model.ID, Valid: true, Placements: placements}
}

func UnknownModel(modelID string) RoutePreview {
	return RoutePreview{Model: modelID, Valid: false, Reason: ReasonUnknownModel}
}

func MissingComputeReason(compute string) Reason {
	return Reason(fmt.Sprintf("%s:%s", ReasonMissingCompute, compute))
}

func routeReason(memoryOK bool, computeOK bool, compute *cluster.ComputeBackend) Reason {
	switch {
	case !memoryOK:
		return ReasonInsufficientMemory
	case !computeOK && compute != nil:
		return MissingComputeReason(string(*compute))
	case !computeOK:
		return ReasonMissingCompute
	default:
		return ReasonCandidate
	}
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
	default:
		return 0
	}
}

func containsStringCapability(caps map[string]any, key string, expected string) bool {
	value, ok := caps[key]
	if !ok {
		return false
	}

	switch items := value.(type) {
	case []string:
		for _, item := range items {
			if item == expected {
				return true
			}
		}
	case []any:
		for _, item := range items {
			if text, ok := item.(string); ok && text == expected {
				return true
			}
		}
	}

	return false
}
