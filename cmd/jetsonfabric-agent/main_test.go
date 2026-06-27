package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const catalogFilePerm os.FileMode = 0o600

func TestAdvertisedBackendsUseAgentProxyURL(t *testing.T) {
	backends := advertisedBackends(
		"http://127.0.0.1:52416",
		"http://127.0.0.1:8080",
		[]string{"qwen2.5-coder-1.5b-q4", "qwen2.5-coder-3b-q4"},
	)
	if len(backends) != 1 {
		t.Fatalf("expected one backend, got %d", len(backends))
	}
	backend := backends[0]
	if backend.ID != cluster.BackendIDLlamaLocal {
		t.Fatalf("unexpected backend ID: %s", backend.ID)
	}
	if backend.Kind != cluster.RuntimeKindLlamaCPP {
		t.Fatalf("unexpected backend kind: %s", backend.Kind)
	}
	if backend.BaseURL != "http://127.0.0.1:52416" {
		t.Fatalf("expected agent proxy URL, got %s", backend.BaseURL)
	}
	if !backend.OpenAICompatible {
		t.Fatal("expected OpenAI-compatible backend")
	}
	expectedModels := []string{"qwen2.5-coder-1.5b-q4", "qwen2.5-coder-3b-q4"}
	if !reflect.DeepEqual(backend.Models, expectedModels) {
		t.Fatalf("unexpected models: %+v", backend.Models)
	}
}

func TestResolveModelArtifactsRequiresAdvertisedModels(t *testing.T) {
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

	artifacts, err := resolveModelArtifacts(path, []string{"qwen2.5-coder-1.5b-q4"})
	if err != nil {
		t.Fatalf("resolve artifacts: %v", err)
	}
	if len(artifacts) != 1 {
		t.Fatalf("expected one artifact, got %d", len(artifacts))
	}
	if artifacts[0].ModelID != "qwen2.5-coder-1.5b-q4" {
		t.Fatalf("unexpected artifact: %+v", artifacts[0])
	}

	_, err = resolveModelArtifacts(path, []string{"missing-model"})
	if err == nil {
		t.Fatal("expected missing artifact error")
	}
	if !strings.Contains(err.Error(), "missing-model") {
		t.Fatalf("expected model ID in error, got %v", err)
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

func TestAdvertisedBackendsRequireAgentAndRuntimeURLs(t *testing.T) {
	tests := []struct {
		name     string
		agentURL string
		llamaURL string
	}{
		{name: "missing agent URL", agentURL: "", llamaURL: "http://127.0.0.1:8080"},
		{name: "missing runtime URL", agentURL: "http://127.0.0.1:52416", llamaURL: ""},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backends := advertisedBackends(test.agentURL, test.llamaURL, []string{"qwen2.5-coder-1.5b-q4"})
			if backends != nil {
				t.Fatalf("expected no backends, got %+v", backends)
			}
		})
	}
}
