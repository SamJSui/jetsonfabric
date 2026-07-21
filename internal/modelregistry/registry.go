package modelregistry

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"

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
	for index := range registry.Models {
		model := &registry.Models[index]
		model.ID = strings.TrimSpace(model.ID)
		model.ArtifactPath = strings.TrimSpace(model.ArtifactPath)
		model.ArtifactSHA256 = strings.ToLower(strings.TrimSpace(model.ArtifactSHA256))
		if model.ID == "" {
			return Registry{}, fmt.Errorf("model registry contains an empty model id")
		}
		if slices.Contains(model.PlacementModes, cluster.ExecutionModePipelineParallel) && model.LayerCount < 2 {
			return Registry{}, fmt.Errorf("model %s advertises pipeline_parallel without at least two layers", model.ID)
		}
		if (model.ArtifactPath == "") != (model.ArtifactSHA256 == "") {
			return Registry{}, fmt.Errorf("model %s must define artifact_path and artifact_sha256 together", model.ID)
		}
		if model.ArtifactSHA256 != "" && !validSHA256(model.ArtifactSHA256) {
			return Registry{}, fmt.Errorf("model %s artifact_sha256 must be a 64-character hexadecimal digest", model.ID)
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

func validSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
