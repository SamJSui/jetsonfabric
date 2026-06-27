package config

import (
	"fmt"
	"path/filepath"
	"time"
)

const (
	DefaultControlHost = "127.0.0.1"
	DefaultControlPort = 52415
	DefaultAgentHost   = "127.0.0.1"
	DefaultAgentPort   = 52416
	DefaultJoinToken   = "dev-token"

	DefaultHeartbeatInterval = 10 * time.Second
)

const (
	configDir          = "configs"
	modelRegistryFile  = "models.example.json"
	modelArtifactsFile = "model-artifacts.example.json"
	dataDir            = "data"
	benchmarksFile     = "benchmarks.jsonl"
	controlURLFormat   = "http://%s:%d"
)

func DefaultControlURL() string {
	return fmt.Sprintf(controlURLFormat, DefaultControlHost, DefaultControlPort)
}

func DefaultAgentURL() string {
	return fmt.Sprintf(controlURLFormat, DefaultAgentHost, DefaultAgentPort)
}

func DefaultModelRegistryPath() string {
	return filepath.Join(configDir, modelRegistryFile)
}

func DefaultModelArtifactsPath() string {
	return filepath.Join(configDir, modelArtifactsFile)
}

func DefaultBenchmarksPath() string {
	return filepath.Join(dataDir, benchmarksFile)
}
