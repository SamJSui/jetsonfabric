package config

import (
	"path/filepath"
)

const (
	configDir         = "configs"
	modelRegistryFile = "models.example.json"
	dataDir           = "data"
	benchmarksFile    = "benchmarks.jsonl"
)

func DefaultModelRegistryPath() string {
	return filepath.Join(configDir, modelRegistryFile)
}

func DefaultBenchmarksPath() string {
	return filepath.Join(dataDir, benchmarksFile)
}
