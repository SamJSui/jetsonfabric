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

func TestAdvertisedEnginesUseAgentProxyURL(t *testing.T) {
	engines := advertisedEngines(
		"http://127.0.0.1:52416",
		cluster.EngineLlamaCPP,
		"http://127.0.0.1:8080",
		"qwen2.5-coder-1.5b-q4",
	)
	if len(engines) != 1 {
		t.Fatalf("expected one engine, got %d", len(engines))
	}

	engine := engines[0]
	if engine.InstanceID != cluster.DefaultEngineInstanceID {
		t.Fatalf("unexpected engine instance ID: %s", engine.InstanceID)
	}
	if engine.Engine != cluster.EngineLlamaCPP {
		t.Fatalf("unexpected engine: %s", engine.Engine)
	}
	if engine.BaseURL != "http://127.0.0.1:52416" {
		t.Fatalf("expected agent proxy URL, got %s", engine.BaseURL)
	}
	if !engine.OpenAICompatible {
		t.Fatal("expected OpenAI-compatible engine")
	}

	expectedModels := []string{"qwen2.5-coder-1.5b-q4"}
	if !reflect.DeepEqual(engine.Models, expectedModels) {
		t.Fatalf("unexpected models: %+v", engine.Models)
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
	if cfg.engine != string(cluster.EngineLlamaCPP) || cfg.engineURL != "http://127.0.0.1:8080" {
		t.Fatalf("expected llama compatibility shortcut to resolve engine fields, got %+v", cfg)
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

func TestValidateAgentConfigRequiresModelEnginePair(t *testing.T) {
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
			name: "engine without model",
			cfg: func() agentConfig {
				cfg := base
				cfg.engine = string(cluster.EngineLlamaCPP)
				cfg.engineURL = "http://127.0.0.1:8080"
				return cfg
			}(),
			want: "--model is required",
		},
		{
			name: "model without engine url",
			cfg: func() agentConfig {
				cfg := base
				cfg.model = "qwen2.5-coder-1.5b-q4"
				return cfg
			}(),
			want: "--engine-url is required",
		},
		{
			name: "once with engine",
			cfg: func() agentConfig {
				cfg := base
				cfg.engine = string(cluster.EngineLlamaCPP)
				cfg.engineURL = "http://127.0.0.1:8080"
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
      "engine": "llama.cpp",
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

func TestAdvertisedEnginesRequireAgentAndEngineURLs(t *testing.T) {
	tests := []struct {
		name      string
		agentURL  string
		engineURL string
		model     string
	}{
		{name: "missing agent URL", agentURL: "", engineURL: "http://127.0.0.1:8080", model: "qwen2.5-coder-1.5b-q4"},
		{name: "missing engine URL", agentURL: "http://127.0.0.1:52416", engineURL: "", model: "qwen2.5-coder-1.5b-q4"},
		{name: "missing model", agentURL: "http://127.0.0.1:52416", engineURL: "http://127.0.0.1:8080", model: ""},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			engines := advertisedEngines(test.agentURL, cluster.EngineLlamaCPP, test.engineURL, test.model)
			if engines != nil {
				t.Fatalf("expected no engines, got %+v", engines)
			}
		})
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
