package node

import (
	"net/url"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/discovery"
)

func TestNormalizeConfigDerivesLocalAdvertiseURL(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.NodeName = "logical-stage-name"
	cfg.Listen = "0.0.0.0:52415"
	cfg.APIURL = ""

	normalized := NormalizeConfig(cfg)
	advertiseURL, err := url.Parse(normalized.APIURL)
	if err != nil {
		t.Fatalf("parse advertise URL %q: %v", normalized.APIURL, err)
	}
	if advertiseURL.Scheme != "http" {
		t.Fatalf("advertise URL scheme = %q, want http", advertiseURL.Scheme)
	}
	if advertiseURL.Port() != "52415" {
		t.Fatalf("advertise URL port = %q, want 52415", advertiseURL.Port())
	}
	hostname := advertiseURL.Hostname()
	if hostname == "" || !strings.HasSuffix(hostname, ".local") {
		t.Fatalf("advertise URL hostname = %q, want nonempty .local hostname", hostname)
	}
	if hostname == cfg.NodeName+".local" {
		t.Fatalf("advertise URL should use the physical hostname, not logical node name %q", cfg.NodeName)
	}
}

func TestNormalizeConfigSupportsDiscoveryNone(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.DiscoveryModes = []string{discovery.ModeNone}

	normalized := NormalizeConfig(cfg)
	if len(normalized.DiscoveryModes) != 0 {
		t.Fatalf("expected discovery disabled, got %+v", normalized.DiscoveryModes)
	}
}

func TestValidateConfigRejectsUnsupportedDiscoveryMode(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.DiscoveryModes = []string{"magic"}

	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("expected unsupported discovery mode error")
	}
}
