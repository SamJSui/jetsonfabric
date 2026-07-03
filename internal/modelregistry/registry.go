package modelregistry

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

type Registry struct {
	Models []cluster.ModelProfile `json:"models"`
}

func Load(path string) (Registry, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return Registry{}, err
	}
	var registry Registry
	if err := json.Unmarshal(content, &registry); err != nil {
		return Registry{}, err
	}
	for _, model := range registry.Models {
		if model.ID == "" {
			return Registry{}, fmt.Errorf("model registry contains an empty model id")
		}
		if slices.Contains(model.PlacementModes, cluster.ExecutionModePipelineParallel) && model.LayerCount < 2 {
			return Registry{}, fmt.Errorf("model %s advertises layer_split without at least two layers", model.ID)
		}
	}
	return registry, nil
}

func (r Registry) Find(id string) (cluster.ModelProfile, bool) {
	for _, model := range r.Models {
		if model.ID == id {
			return model, true
		}
	}
	return cluster.ModelProfile{}, false
}
