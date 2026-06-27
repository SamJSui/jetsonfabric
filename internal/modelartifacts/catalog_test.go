package modelartifacts

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const catalogFilePerm os.FileMode = 0o600

func TestLoadValidatesAndFindsArtifacts(t *testing.T) {
	path := writeCatalog(t, `{
  "artifacts": [
    {
      "model_id": "qwen2.5-coder-1.5b-q4",
      "runtime": "llama.cpp",
      "source_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
      "local_path": ".cache/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    }
  ]
}`)

	catalog, err := Load(path)
	if err != nil {
		t.Fatalf("load catalog: %v", err)
	}
	artifact, ok := catalog.Find("qwen2.5-coder-1.5b-q4")
	if !ok {
		t.Fatal("expected artifact")
	}
	if artifact.Runtime != cluster.RuntimeKindLlamaCPP {
		t.Fatalf("unexpected runtime: %s", artifact.Runtime)
	}
	if artifact.LocalPath != ".cache/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" {
		t.Fatalf("unexpected local path: %s", artifact.LocalPath)
	}
}

func TestLoadRejectsInvalidSourceURL(t *testing.T) {
	path := writeCatalog(t, `{
  "artifacts": [
    {
      "model_id": "qwen2.5-coder-1.5b-q4",
      "runtime": "llama.cpp",
      "source_url": "not-a-url",
      "local_path": ".cache/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    }
  ]
}`)

	if _, err := Load(path); err == nil {
		t.Fatal("expected invalid source URL error")
	}
}

func writeCatalog(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model-artifacts.json")
	if err := os.WriteFile(path, []byte(content), catalogFilePerm); err != nil {
		t.Fatalf("write catalog: %v", err)
	}
	return path
}
