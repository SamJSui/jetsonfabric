package node

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/config"
	"github.com/SamJSui/jetsonfabric/internal/discovery"
)

const (
	DefaultClusterID         = "default"
	DefaultDataDir           = ".cache/jetsonfabric"
	DefaultDiscoveryInterval = 10 * time.Second
	DefaultStaleAfter        = 30 * time.Second
	DefaultControlPriority   = 10
)

var defaultDiscoveryModes = []string{discovery.ModeStatic, discovery.ModeMDNS}

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
	DiscoveryModes    []string
	DiscoveryInterval time.Duration
	StaleAfter        time.Duration

	MDNSService       string
	MDNSDomain        string
	MDNSBrowseTimeout time.Duration

	JoinToken      string
	ModelsPath     string
	BenchmarksPath string
}

func DefaultConfigValue() Config {
	return Config{
		ClusterID:         DefaultClusterID,
		Listen:            config.DefaultControlListen(),
		APIURL:            "",
		DataDir:           DefaultDataDir,
		RuntimeURL:        "http://127.0.0.1:9090",
		Engine:            cluster.EngineJetsonFabric,
		ControlEligible:   true,
		ControlPriority:   DefaultControlPriority,
		Seeds:             nil,
		DiscoveryModes:    append([]string(nil), defaultDiscoveryModes...),
		DiscoveryInterval: DefaultDiscoveryInterval,
		StaleAfter:        DefaultStaleAfter,
		MDNSService:       discovery.DefaultMDNSService,
		MDNSDomain:        discovery.DefaultMDNSDomain,
		MDNSBrowseTimeout: discovery.DefaultMDNSBrowseTimeout,
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
	cfg.DiscoveryModes = normalizeDiscoveryModes(cfg.DiscoveryModes)
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
	if cfg.APIURL == "" {
		cfg.APIURL = defaultAdvertiseURL(cfg.NodeName, cfg.Listen)
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
	if cfg.APIURL == "" {
		return fmt.Errorf("--advertise-url could not be derived; set it explicitly")
	}
	if cfg.DataDir == "" {
		return fmt.Errorf("--data-dir is required")
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

func (cfg Config) DiscoveryEnabled(mode string) bool {
	mode = strings.ToLower(strings.TrimSpace(mode))
	for _, configured := range cfg.DiscoveryModes {
		if configured == mode {
			return true
		}
	}
	return false
}

func (cfg Config) AdvertisePort() int {
	parsed, err := url.Parse(cfg.APIURL)
	if err == nil && parsed.Port() != "" {
		port, _ := strconv.Atoi(parsed.Port())
		if port > 0 {
			return port
		}
	}
	_, portText, err := net.SplitHostPort(cfg.Listen)
	if err != nil {
		return 0
	}
	port, _ := strconv.Atoi(portText)
	return port
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

func defaultAdvertiseURL(nodeName string, listen string) string {
	_, port, err := net.SplitHostPort(listen)
	if err != nil || port == "" {
		return ""
	}
	host := strings.TrimSpace(nodeName)
	if host == "" {
		host, _ = os.Hostname()
	}
	host = strings.TrimSpace(strings.TrimSuffix(host, "."))
	if host == "" {
		host = "127.0.0.1"
	} else if !strings.Contains(host, ".") && host != "localhost" {
		host += ".local"
	}
	return "http://" + host + ":" + port
}
