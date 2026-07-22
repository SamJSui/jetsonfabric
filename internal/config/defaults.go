package config

import (
	"fmt"
	"path/filepath"
)

const (
	DefaultNodeHost = "127.0.0.1"
	DefaultNodePort = 52415
)

const (
	configDir         = "configs"
	modelRegistryFile = "models.example.json"
	dataDir           = "data"
	benchmarksFile    = "benchmarks.jsonl"
	urlFormat         = "http://%s:%d"
	listenFormat      = "%s:%d"
)

func DefaultNodeListen() string {
	return fmt.Sprintf(listenFormat, DefaultNodeHost, DefaultNodePort)
}

func DefaultNodeURL() string {
	return fmt.Sprintf(urlFormat, DefaultNodeHost, DefaultNodePort)
}

func DefaultModelRegistryPath() string {
	return filepath.Join(configDir, modelRegistryFile)
}

func DefaultBenchmarksPath() string {
	return filepath.Join(dataDir, benchmarksFile)
}
