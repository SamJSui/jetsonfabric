package config

import (
	"fmt"
	"path/filepath"
	"time"
)

const (
	DefaultControlHost = "127.0.0.1"
	DefaultControlPort = 52415
	DefaultJoinToken   = "dev-token"

	DefaultHeartbeatInterval = 10 * time.Second
)

const (
	configDir         = "configs"
	modelRegistryFile = "models.example.json"
	dataDir           = "data"
	benchmarksFile    = "benchmarks.jsonl"
	controlURLFormat  = "http://%s:%d"
)

func DefaultControlURL() string {
	return fmt.Sprintf(controlURLFormat, DefaultControlHost, DefaultControlPort)
}

func DefaultModelRegistryPath() string {
	return filepath.Join(configDir, modelRegistryFile)
}

func DefaultBenchmarksPath() string {
	return filepath.Join(dataDir, benchmarksFile)
}
