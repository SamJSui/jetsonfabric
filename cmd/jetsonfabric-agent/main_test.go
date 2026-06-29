package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const catalogFilePerm os.FileMode = 0o600

func TestAdvertisedBackendsUseAgentProxyURL(t *testing.T) {
	backends := advertisedBackends(
		"http://127.0.0.1:52416",
		"http://127.0.0.1:8080",
		"qwen2.5-coder-1.5b-q4",
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
	expectedModels := []string{"qwen2.5-coder-1.5b-q4"}
	if !reflect.DeepEqual(backend.Models, expectedModels) {
		t.Fatalf("unexpected models: %+v", backend.Models)
	}
}

func TestParseAgentConfigUsesExplicitNodeNameAndListen(t *testing.T) {
	cfg, err := parseAgentConfig([]string{
		"--control-url", "http://control:52415",
		"--node-name", "dopey",
		"--listen", "0.0.0.0:52416",
		"--advertise-url", "http://dopey:52416",
		"--llama-url", "http://127.0.0.1:8080",
		"--model", "qwen2.5-coder-1.5b-q4",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.nodeName != "dopey" || cfg.listen != "0.0.0.0:52416" {
		t.Fatalf("unexpected config: %+v", cfg)
	}
	if err := validateAgentConfig(cfg); err != nil {
		t.Fatalf("validate config: %v", err)
	}
}

func TestParseAgentConfigDefaultsNodeNameFromHostname(t *testing.T) {
	cfg, err := parseAgentConfig([]string{"--control-url", "http://control:52415", "--once"})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if strings.TrimSpace(cfg.nodeName) == "" {
		t.Fatalf("expected hostname fallback, got %+v", cfg)
	}
}

func TestValidateAgentConfigRequiresModelRuntimePair(t *testing.T) {
	base := agentConfig{
		controlURL:   "http://control:52415",
		joinToken:    "dev-token",
		nodeName:     "dopey",
		listen:       "127.0.0.1:52416",
		advertiseURL: "http://dopey:52416",
		interval:     time.Second,
	}
	tests := []struct {
		name string
		cfg  agentConfig
		want string
	}{
		{
			name: "runtime without model",
			cfg: func() agentConfig {
				cfg := base
				cfg.llamaURL = "http://127.0.0.1:8080"
				return cfg
			}(),
			want: "--model is required",
		},
		{
			name: "model without runtime",
			cfg: func() agentConfig {
				cfg := base
				cfg.model = "qwen2.5-coder-1.5b-q4"
				return cfg
			}(),
			want: "--llama-url is required",
		},
		{
			name: "once with runtime",
			cfg: func() agentConfig {
				cfg := base
				cfg.llamaURL = "http://127.0.0.1:8080"
				cfg.model = "qwen2.5-coder-1.5b-q4"
				cfg.once = true
				return cfg
			}(),
			want: "--once cannot be used",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateAgentConfig(test.cfg)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
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
		model    string
	}{
		{name: "missing agent URL", agentURL: "", llamaURL: "http://127.0.0.1:8080", model: "qwen2.5-coder-1.5b-q4"},
		{name: "missing runtime URL", agentURL: "http://127.0.0.1:52416", llamaURL: "", model: "qwen2.5-coder-1.5b-q4"},
		{name: "missing model", agentURL: "http://127.0.0.1:52416", llamaURL: "http://127.0.0.1:8080", model: ""},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backends := advertisedBackends(test.agentURL, test.llamaURL, test.model)
			if backends != nil {
				t.Fatalf("expected no backends, got %+v", backends)
			}
		})
	}
}
