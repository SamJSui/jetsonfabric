package routing

import (
	"fmt"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
)

type Reason string

const (
	ReasonUnknownModel       Reason = "unknown_model"
	ReasonInsufficientMemory Reason = "insufficient_memory"
	ReasonMissingAccelerator Reason = "missing_accelerator"
	ReasonCandidate          Reason = "candidate"
)

type PlacementPreview struct {
	NodeID        string `json:"node_id"`
	Valid         bool   `json:"valid"`
	MemoryOK      bool   `json:"memory_ok"`
	AcceleratorOK bool   `json:"accelerator_ok"`
	Reason        Reason `json:"reason"`
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
		acceleratorOK := true
		if model.PreferredAccelerator != nil && *model.PreferredAccelerator != "" {
			acceleratorOK = containsStringCapability(node.Capabilities, cluster.CapabilityAccelerators, *model.PreferredAccelerator)
		}
		memoryOK := memory >= model.MinMemoryGB
		placements = append(placements, PlacementPreview{
			NodeID:        node.NodeID,
			Valid:         memoryOK && acceleratorOK,
			MemoryOK:      memoryOK,
			AcceleratorOK: acceleratorOK,
			Reason:        routeReason(memoryOK, acceleratorOK, model.PreferredAccelerator),
		})
	}
	return RoutePreview{Model: model.ID, Valid: true, Placements: placements}
}

func UnknownModel(modelID string) RoutePreview {
	return RoutePreview{Model: modelID, Valid: false, Reason: ReasonUnknownModel}
}

func MissingAcceleratorReason(accelerator string) Reason {
	return Reason(fmt.Sprintf("%s:%s", ReasonMissingAccelerator, accelerator))
}

func routeReason(memoryOK bool, acceleratorOK bool, accelerator *string) Reason {
	switch {
	case !memoryOK:
		return ReasonInsufficientMemory
	case !acceleratorOK && accelerator != nil:
		return MissingAcceleratorReason(*accelerator)
	case !acceleratorOK:
		return ReasonMissingAccelerator
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
	items, ok := value.([]any)
	if !ok {
		return false
	}
	for _, item := range items {
		if text, ok := item.(string); ok && text == expected {
			return true
		}
	}
	return false
}
