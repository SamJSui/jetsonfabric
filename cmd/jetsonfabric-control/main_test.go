package main

import (
	"strings"
	"testing"
)

func TestParseControlConfigUsesListen(t *testing.T) {
	cfg, err := parseControlConfig([]string{
		"--listen", "0.0.0.0:52415",
		"--join-token", "test-token",
		"--models", "configs/models.example.json",
		"--benchmarks", "data/test-benchmarks.jsonl",
	})
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.listen != "0.0.0.0:52415" {
		t.Fatalf("unexpected listen address: %+v", cfg)
	}
	if err := validateControlConfig(cfg); err != nil {
		t.Fatalf("validate config: %v", err)
	}
}

func TestValidateControlConfigRequiresListen(t *testing.T) {
	cfg := controlConfig{
		modelsPath:     "configs/models.example.json",
		benchmarksPath: "data/test-benchmarks.jsonl",
	}
	err := validateControlConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "--listen is required") {
		t.Fatalf("expected listen validation error, got %v", err)
	}
}
