package node

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/discovery"
)

func TestNormalizeConfigDerivesLocalAdvertiseURL(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.NodeName = "dopey"
	cfg.Listen = "0.0.0.0:52415"
	cfg.APIURL = ""

	normalized := NormalizeConfig(cfg)
	if normalized.APIURL != "http://dopey.local:52415" {
		t.Fatalf("unexpected advertise URL: %s", normalized.APIURL)
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
