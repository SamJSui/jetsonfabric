package node

import (
	"fmt"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/config"
)

const (
	DefaultClusterID         = "default"
	DefaultDataDir           = ".cache/jetsonfabric"
	DefaultDiscoveryInterval = 10 * time.Second
	DefaultStaleAfter        = 30 * time.Second
	DefaultControlPriority   = 10
)

type Config struct {
	ClusterID string
	NodeName  string
	Listen    string
	APIURL    string
	DataDir   string

	RuntimeURL string
	Engine     cluster.Engine
	Model      string

	ControlEligible bool
	ControlPriority int

	Seeds             []string
	DiscoveryInterval time.Duration
	StaleAfter        time.Duration

	JoinToken      string
	ModelsPath     string
	BenchmarksPath string
}

func DefaultConfigValue() Config {
	return Config{
		ClusterID:         DefaultClusterID,
		Listen:            config.DefaultControlListen(),
		APIURL:            config.DefaultControlURL(),
		DataDir:           DefaultDataDir,
		RuntimeURL:        "http://127.0.0.1:9090",
		Engine:            cluster.EngineJetsonFabric,
		ControlEligible:   true,
		ControlPriority:   DefaultControlPriority,
		DiscoveryInterval: DefaultDiscoveryInterval,
		StaleAfter:        DefaultStaleAfter,
		JoinToken:         config.DefaultJoinToken,
		ModelsPath:        config.DefaultModelRegistryPath(),
		BenchmarksPath:    config.DefaultBenchmarksPath(),
	}
}

func NormalizeConfig(cfg Config) Config {
	cfg.ClusterID = strings.TrimSpace(cfg.ClusterID)
	cfg.NodeName = strings.TrimSpace(cfg.NodeName)
	cfg.Listen = strings.TrimSpace(cfg.Listen)
	cfg.APIURL = strings.TrimSpace(cfg.APIURL)
	cfg.DataDir = strings.TrimSpace(cfg.DataDir)
	cfg.RuntimeURL = strings.TrimSpace(cfg.RuntimeURL)
	cfg.Model = strings.TrimSpace(cfg.Model)
	cfg.JoinToken = strings.TrimSpace(cfg.JoinToken)
	cfg.ModelsPath = strings.TrimSpace(cfg.ModelsPath)
	cfg.BenchmarksPath = strings.TrimSpace(cfg.BenchmarksPath)
	cfg.Seeds = normalizeStrings(cfg.Seeds)
	return cfg
}

func ValidateConfig(cfg Config) error {
	if cfg.ClusterID == "" {
		return fmt.Errorf("--cluster-id is required")
	}
	if cfg.Listen == "" {
		return fmt.Errorf("--listen is required")
	}
	if cfg.APIURL == "" {
		return fmt.Errorf("--advertise-url is required")
	}
	if cfg.DataDir == "" {
		return fmt.Errorf("--data-dir is required")
	}
	if cfg.DiscoveryInterval <= 0 {
		return fmt.Errorf("--discovery-interval must be greater than zero")
	}
	if cfg.StaleAfter <= 0 {
		return fmt.Errorf("--stale-after must be greater than zero")
	}
	if cfg.ModelsPath == "" {
		return fmt.Errorf("--models is required")
	}
	if cfg.BenchmarksPath == "" {
		return fmt.Errorf("--benchmarks is required")
	}
	return nil
}

func normalizeStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}
