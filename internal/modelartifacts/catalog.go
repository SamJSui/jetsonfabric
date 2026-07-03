package modelartifacts

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

type Catalog struct {
	Artifacts []Artifact `json:"artifacts"`
}

type Artifact struct {
	ModelID        string         `json:"model_id"`
	Engine         cluster.Engine `json:"engine"`
	SourceURL      string         `json:"source_url"`
	LocalPath      string         `json:"local_path"`
	ExpectedSHA256 string         `json:"expected_sha256,omitempty"`
}

func Load(path string) (Catalog, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return Catalog{}, err
	}
	var catalog Catalog
	if err := json.Unmarshal(content, &catalog); err != nil {
		return Catalog{}, err
	}
	for _, artifact := range catalog.Artifacts {
		if artifact.ModelID == "" {
			return Catalog{}, fmt.Errorf("model artifact catalog contains an empty model id")
		}
		if artifact.Engine == "" {
			return Catalog{}, fmt.Errorf("model artifact %q contains an empty engine", artifact.ModelID)
		}
		if artifact.SourceURL == "" {
			return Catalog{}, fmt.Errorf("model artifact %q contains an empty source url", artifact.ModelID)
		}
		if artifact.LocalPath == "" {
			return Catalog{}, fmt.Errorf("model artifact %q contains an empty local path", artifact.ModelID)
		}
		parsed, err := url.Parse(artifact.SourceURL)
		if err != nil {
			return Catalog{}, fmt.Errorf("parse source url for model artifact %q: %w", artifact.ModelID, err)
		}
		if parsed.Scheme == "" || parsed.Host == "" {
			return Catalog{}, fmt.Errorf("model artifact %q source url must include scheme and host", artifact.ModelID)
		}
	}
	return catalog, nil
}

func (c Catalog) Find(modelID string) (Artifact, bool) {
	for _, artifact := range c.Artifacts {
		if artifact.ModelID == modelID {
			return artifact, true
		}
	}
	return Artifact{}, false
}
