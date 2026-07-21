package node

import (
	"fmt"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/config"
	"github.com/SamJSui/jetsonfabric/internal/discovery"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

const (
	DefaultClusterID         = "home-lab"
	DefaultNodeListen        = "0.0.0.0:0"
	DefaultDiscoveryInterval = 10 * time.Second
	DefaultStaleAfter        = 30 * time.Second
	DefaultLeaderPreference  = 0
	ClusterTokenEnv          = "JETSONFABRIC_CLUSTER_TOKEN"

	AutoRuntimeURL                 = "auto"
	DefaultRuntimeBin              = "dist/jetsonfabric-runtime-worker"
	DefaultRuntimeListen           = "127.0.0.1:0"
	DefaultRuntimeComputeBackend   = "cuda"
	DefaultRuntimeMode             = "pipeline_parallel"
	DefaultRuntimeCtxSize          = 4096
	DefaultRuntimeNGPULayers       = 999
	DefaultRuntimeThreads          = 0
	DefaultRuntimeRevision         = "dev"
	DefaultRuntimeLlamaCPPRevision = "dev"

	DefaultStageIndex = 0
	DefaultStageCount = 1
	DefaultLayerStart = 0
	DefaultLayerEnd   = 28
)

var defaultDiscoveryModes = []string{discovery.ModeMDNS}

type Config struct {
	ClusterID    string
	ClusterToken string
	NodeName     string
	Listen       string
	APIURL       string
	DataDir      string

	RuntimeURL string
	RuntimeBin string

	Engine                  cluster.Engine
	Model                   string
	ModelPath               string
	RuntimeListen           string
	RuntimeComputeBackend   string
	RuntimeMode             string
	RuntimeCtxSize          int
	RuntimeNGPULayers       int
	RuntimeThreads          int
	RuntimeStartIdle        bool
	RuntimeRevision         string
	RuntimeLlamaCPPRevision string
	RuntimeCUDAActive       bool

	StageIndex int
	StageCount int
	LayerStart int
	LayerEnd   int

	Role             membership.NodeRole
	LeaderPreference int

	Seeds             []string
	DiscoveryModes    []string
	DiscoveryInterval time.Duration
	StaleAfter        time.Duration

	MDNSService       string
	MDNSDomain        string
	MDNSBrowseTimeout time.Duration

	ModelsPath     string
	BenchmarksPath string
}

func DefaultConfigValue() Config {
	return Config{
		ClusterID:               DefaultClusterID,
		Listen:                  DefaultNodeListen,
		APIURL:                  "",
		DataDir:                 "",
		RuntimeURL:              AutoRuntimeURL,
		RuntimeBin:              DefaultRuntimeBin,
		Engine:                  cluster.EngineLlamaCPP,
		Model:                   "qwen2.5-coder-1.5b-q4",
		ModelPath:               "",
		RuntimeListen:           DefaultRuntimeListen,
		RuntimeComputeBackend:   DefaultRuntimeComputeBackend,
		RuntimeMode:             DefaultRuntimeMode,
		RuntimeCtxSize:          DefaultRuntimeCtxSize,
		RuntimeNGPULayers:       DefaultRuntimeNGPULayers,
		RuntimeThreads:          DefaultRuntimeThreads,
		RuntimeRevision:         DefaultRuntimeRevision,
		RuntimeLlamaCPPRevision: DefaultRuntimeLlamaCPPRevision,
		StageIndex:              DefaultStageIndex,
		StageCount:              DefaultStageCount,
		LayerStart:              DefaultLayerStart,
		LayerEnd:                DefaultLayerEnd,
		Role:                    membership.NodeRoleAuto,
		LeaderPreference:        DefaultLeaderPreference,
		Seeds:                   nil,
		DiscoveryModes:          append([]string(nil), defaultDiscoveryModes...),
		DiscoveryInterval:       DefaultDiscoveryInterval,
		StaleAfter:              DefaultStaleAfter,
		MDNSService:             discovery.DefaultMDNSService,
		MDNSDomain:              discovery.DefaultMDNSDomain,
		MDNSBrowseTimeout:       discovery.DefaultMDNSBrowseTimeout,
		ModelsPath:              config.DefaultModelRegistryPath(),
		BenchmarksPath:          config.DefaultBenchmarksPath(),
	}
}

func NormalizeConfig(cfg Config) Config {
	cfg = normalizeStringsInConfig(cfg)
	cfg.Seeds = normalizeStrings(cfg.Seeds)
	cfg.DiscoveryModes = normalizeDiscoveryModes(cfg.DiscoveryModes)
	cfg.Role = resolveNodeRole(cfg.Role)
	cfg = normalizeMDNSConfig(cfg)
	cfg = normalizeRuntimeConfig(cfg)

	// Only works for explicit non-zero ports.
	// For 0.0.0.0:0, APIURL is derived after net.Listen in app.Run.
	if cfg.APIURL == "" {
		cfg.APIURL = defaultAdvertiseURL(cfg.Listen)
	}

	return cfg
}

func normalizeStringsInConfig(cfg Config) Config {
	cfg.ClusterID = strings.TrimSpace(cfg.ClusterID)
	cfg.ClusterToken = strings.TrimSpace(cfg.ClusterToken)
	cfg.NodeName = strings.TrimSpace(cfg.NodeName)
	cfg.Listen = strings.TrimSpace(cfg.Listen)
	cfg.APIURL = strings.TrimSpace(cfg.APIURL)
	cfg.DataDir = strings.TrimSpace(cfg.DataDir)

	cfg.RuntimeURL = strings.TrimSpace(cfg.RuntimeURL)
	cfg.RuntimeBin = strings.TrimSpace(cfg.RuntimeBin)
	cfg.Model = strings.TrimSpace(cfg.Model)
	cfg.ModelPath = strings.TrimSpace(cfg.ModelPath)
	cfg.RuntimeListen = strings.TrimSpace(cfg.RuntimeListen)
	cfg.RuntimeComputeBackend = strings.TrimSpace(cfg.RuntimeComputeBackend)
	cfg.RuntimeMode = strings.TrimSpace(cfg.RuntimeMode)
	cfg.RuntimeRevision = strings.TrimSpace(cfg.RuntimeRevision)
	cfg.RuntimeLlamaCPPRevision = strings.TrimSpace(cfg.RuntimeLlamaCPPRevision)

	cfg.ModelsPath = strings.TrimSpace(cfg.ModelsPath)
	cfg.BenchmarksPath = strings.TrimSpace(cfg.BenchmarksPath)
	return cfg
}

func normalizeRuntimeConfig(cfg Config) Config {
	if cfg.RuntimeURL == "" {
		cfg.RuntimeURL = AutoRuntimeURL
	}
	if cfg.RuntimeBin == "" {
		cfg.RuntimeBin = DefaultRuntimeBin
	}
	if cfg.RuntimeListen == "" {
		cfg.RuntimeListen = DefaultRuntimeListen
	}
	if cfg.Engine == "" {
		cfg.Engine = cluster.EngineLlamaCPP
	}
	if cfg.RuntimeComputeBackend == "" {
		cfg.RuntimeComputeBackend = DefaultRuntimeComputeBackend
	}
	if cfg.RuntimeMode == "" {
		cfg.RuntimeMode = DefaultRuntimeMode
	}
	if cfg.RuntimeCtxSize <= 0 {
		cfg.RuntimeCtxSize = DefaultRuntimeCtxSize
	}
	if cfg.RuntimeNGPULayers == 0 {
		cfg.RuntimeNGPULayers = DefaultRuntimeNGPULayers
	}
	if cfg.RuntimeRevision == "" {
		cfg.RuntimeRevision = DefaultRuntimeRevision
	}
	if cfg.RuntimeLlamaCPPRevision == "" {
		cfg.RuntimeLlamaCPPRevision = DefaultRuntimeLlamaCPPRevision
	}
	if cfg.StageCount <= 0 {
		cfg.StageCount = DefaultStageCount
	}
	if cfg.LayerEnd <= cfg.LayerStart {
		cfg.LayerStart = DefaultLayerStart
		cfg.LayerEnd = DefaultLayerEnd
	}
	return cfg
}

func normalizeMDNSConfig(cfg Config) Config {
	cfg.MDNSService = strings.TrimSpace(cfg.MDNSService)
	cfg.MDNSDomain = strings.TrimSpace(cfg.MDNSDomain)
	if cfg.MDNSService == "" {
		cfg.MDNSService = discovery.DefaultMDNSService
	}
	if cfg.MDNSDomain == "" {
		cfg.MDNSDomain = discovery.DefaultMDNSDomain
	}
	if cfg.MDNSBrowseTimeout <= 0 {
		cfg.MDNSBrowseTimeout = discovery.DefaultMDNSBrowseTimeout
	}
	return cfg
}

func ValidateConfig(cfg Config) error {
	if cfg.ClusterID == "" {
		return fmt.Errorf("--cluster-id is required")
	}
	if cfg.Listen == "" {
		return fmt.Errorf("--listen is required")
	}
	if cfg.NodeName == "" {
		return fmt.Errorf("--node-name could not be derived")
	}
	if cfg.DataDir == "" {
		return fmt.Errorf("--data-dir could not be derived")
	}
	if cfg.Model == "" {
		return fmt.Errorf("--model is required")
	}
	if cfg.Engine == "" {
		return fmt.Errorf("--engine is required")
	}
	if cfg.RuntimeRevision == "" {
		return fmt.Errorf("--runtime-revision is required")
	}
	if cfg.Engine == cluster.EngineLlamaCPP && cfg.RuntimeLlamaCPPRevision == "" {
		return fmt.Errorf("--runtime-llama-cpp-revision is required for llama.cpp")
	}
	if cfg.RuntimeAuto() {
		if cfg.RuntimeBin == "" {
			return fmt.Errorf("--runtime-bin is required when --runtime-url=auto")
		}
		if !cfg.RuntimeStartIdle && cfg.ModelPath == "" {
			return fmt.Errorf("--model-path is required when --runtime-url=auto unless --runtime-idle is set")
		}
	}
	if err := validateDiscoveryModes(cfg.DiscoveryModes); err != nil {
		return err
	}
	if cfg.DiscoveryInterval <= 0 {
		return fmt.Errorf("--discovery-interval must be greater than zero")
	}
	if cfg.StaleAfter <= 0 {
		return fmt.Errorf("--stale-after must be greater than zero")
	}
	if cfg.MDNSBrowseTimeout <= 0 {
		return fmt.Errorf("--mdns-browse-timeout must be greater than zero")
	}
	if cfg.ModelsPath == "" {
		return fmt.Errorf("--models is required")
	}
	if cfg.BenchmarksPath == "" {
		return fmt.Errorf("--benchmarks is required")
	}
	return nil
}

func (cfg Config) RuntimeAuto() bool {
	return strings.EqualFold(strings.TrimSpace(cfg.RuntimeURL), AutoRuntimeURL)
}

func (cfg Config) DiscoveryEnabled(mode string) bool {
	mode = strings.ToLower(strings.TrimSpace(mode))
	for _, configured := range cfg.DiscoveryModes {
		if configured == mode {
			return true
		}
	}
	return false
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

func normalizeDiscoveryModes(values []string) []string {
	if len(values) == 0 {
		return append([]string(nil), defaultDiscoveryModes...)
	}
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" {
			continue
		}
		if value == discovery.ModeNone {
			return nil
		}
		if seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	if len(out) == 0 {
		return append([]string(nil), defaultDiscoveryModes...)
	}
	return out
}

func validateDiscoveryModes(values []string) error {
	for _, value := range values {
		switch value {
		case discovery.ModeStatic, discovery.ModeMDNS:
		default:
			return fmt.Errorf("unsupported discovery mode %q", value)
		}
	}
	return nil
}
