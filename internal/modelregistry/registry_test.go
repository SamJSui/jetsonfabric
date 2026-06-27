package modelregistry

import (
	"os"
	"path/filepath"
	"testing"
)

const registryFilePerm os.FileMode = 0o644

func TestLoadRejectsLayerSplitWithoutLayerCount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.json")
	content := []byte(`{
  "models": [
    {
      "id": "qwen",
      "family": "llm",
      "runtime": "llama.cpp",
      "min_memory_gb": 3,
      "preferred_accelerator": null,
      "placement_modes": ["layer_split"]
    }
  ]
}`)
	if err := os.WriteFile(path, content, registryFilePerm); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("expected invalid registry to fail")
	}
}
