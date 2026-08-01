package node

import (
	"net/url"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
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

func TestNormalizeConfigDefaultsRuntimeStageTransport(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.RuntimeStageTransport = ""

	normalized := NormalizeConfig(cfg)

	if normalized.RuntimeStageTransport != DefaultRuntimeStageTransport {
		t.Fatalf(
			"runtime stage transport = %q, want %q",
			normalized.RuntimeStageTransport,
			DefaultRuntimeStageTransport,
		)
	}
}

func TestNormalizeConfigDefaultsRuntimeActivationEncoding(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.RuntimeActivationEncoding = ""

	normalized := NormalizeConfig(cfg)

	if normalized.RuntimeActivationEncoding != DefaultRuntimeActivationEncoding {
		t.Fatalf(
			"runtime activation encoding = %q, want %q",
			normalized.RuntimeActivationEncoding,
			DefaultRuntimeActivationEncoding,
		)
	}
}

func TestNormalizeConfigDefaultsRuntimeKVCacheType(t *testing.T) {
	cfg := DefaultConfigValue()
	cfg.RuntimeKVCacheType = ""

	normalized := NormalizeConfig(cfg)

	if normalized.RuntimeKVCacheType != DefaultRuntimeKVCacheType {
		t.Fatalf(
			"runtime KV cache type = %q, want %q",
			normalized.RuntimeKVCacheType,
			DefaultRuntimeKVCacheType,
		)
	}
}

func TestRuntimeArgsPassConfiguredStrategiesAndWorkerLimits(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.RuntimeStageTransport = "test_transport"
	cfg.RuntimeActivationEncoding = cluster.ActivationEncodingF16
	cfg.RuntimeKVCacheType = cluster.KVCacheTypeQ8_0
	cfg.RuntimeUBatchSize = 128
	cfg.RuntimeHTTPWorkers = 3
	cfg.RuntimeParallelSessions = 4
	cfg.RuntimeDecodeBatchSize = 2
	cfg.RuntimeSpeculativeDraft = "none"
	cfg.RuntimeSpeculativeMax = 6

	args := runtimeArgs(cfg, "127.0.0.1:9090")

	hasTransport := false
	hasActivationEncoding := false
	hasKVCacheType := false
	hasUBatchSize := false
	hasHTTPWorkers := false
	hasParallelSessions := false
	hasDecodeBatchSize := false
	hasSpeculativeDraft := false
	hasSpeculativeMax := false
	for index := 0; index+1 < len(args); index++ {
		if args[index] == "--stage-transport" && args[index+1] == "test_transport" {
			hasTransport = true
		}
		if args[index] == "--activation-encoding" && args[index+1] == cluster.ActivationEncodingF16 {
			hasActivationEncoding = true
		}
		if args[index] == "--kv-cache-type" && args[index+1] == cluster.KVCacheTypeQ8_0 {
			hasKVCacheType = true
		}
		if args[index] == "--ubatch-size" && args[index+1] == "128" {
			hasUBatchSize = true
		}
		if args[index] == "--http-workers" && args[index+1] == "3" {
			hasHTTPWorkers = true
		}
		if args[index] == "--parallel-sessions" && args[index+1] == "4" {
			hasParallelSessions = true
		}
		if args[index] == "--decode-batch-size" && args[index+1] == "2" {
			hasDecodeBatchSize = true
		}
		if args[index] == "--speculative-draft" && args[index+1] == "none" {
			hasSpeculativeDraft = true
		}
		if args[index] == "--speculative-max-tokens" && args[index+1] == "6" {
			hasSpeculativeMax = true
		}
	}
	if !hasTransport || !hasActivationEncoding || !hasKVCacheType || !hasUBatchSize ||
		!hasHTTPWorkers || !hasParallelSessions || !hasDecodeBatchSize ||
		!hasSpeculativeDraft || !hasSpeculativeMax {
		t.Fatalf("runtime args omitted strategy or worker-limit configuration: %v", args)
	}
}

func TestValidateConfigRejectsIncompatibleSpeculativeDecoding(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.RuntimeSpeculativeDraft = "prompt_lookup"
	cfg.RuntimeDecodeBatchSize = 2
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("ValidateConfig accepted speculative decoding with continuous batching")
	}

	cfg = NormalizeConfig(DefaultConfigValue())
	cfg.RuntimeSpeculativeDraft = "unknown"
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("ValidateConfig accepted an unknown speculative draft strategy")
	}
}

func TestValidateConfigRejectsInvalidContinuousBatching(t *testing.T) {
	tests := []struct {
		name             string
		httpWorkers      int
		parallelSessions int
		decodeBatchSize  int
	}{
		{name: "batch exceeds workers", httpWorkers: 1, parallelSessions: 2, decodeBatchSize: 2},
		{name: "batch exceeds sessions", httpWorkers: 4, parallelSessions: 1, decodeBatchSize: 2},
		{name: "no session capacity", httpWorkers: 2, parallelSessions: -1, decodeBatchSize: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := NormalizeConfig(DefaultConfigValue())
			cfg.RuntimeHTTPWorkers = test.httpWorkers
			cfg.RuntimeParallelSessions = test.parallelSessions
			cfg.RuntimeDecodeBatchSize = test.decodeBatchSize
			if err := ValidateConfig(cfg); err == nil {
				t.Fatal("ValidateConfig accepted invalid continuous batching limits")
			}
		})
	}
}

func TestValidateConfigRejectsInvalidRuntimeKVCacheType(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.RuntimeKVCacheType = "q2_k"

	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("ValidateConfig accepted an unsupported runtime KV cache type")
	}
}

func TestValidateConfigRejectsInvalidRuntimeHTTPWorkers(t *testing.T) {
	for _, workers := range []int{-1, MaxRuntimeHTTPWorkers + 1} {
		cfg := NormalizeConfig(DefaultConfigValue())
		cfg.RuntimeHTTPWorkers = workers

		if err := ValidateConfig(cfg); err == nil {
			t.Fatalf("ValidateConfig accepted %d runtime HTTP workers", workers)
		}
	}
}

func TestValidateConfigRequiresTokenForExternallyBoundRuntime(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.NodeName = "dopey"
	cfg.DataDir = t.TempDir()
	cfg.RuntimeStartIdle = true
	cfg.RuntimeListen = "0.0.0.0:9090"
	cfg.ClusterToken = ""
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("ValidateConfig accepted an externally bound runtime without a cluster token")
	}

	cfg.RuntimeListen = "127.0.0.1:9090"
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("ValidateConfig rejected an unauthenticated loopback runtime: %v", err)
	}

	cfg.RuntimeListen = "0.0.0.0:9090"
	cfg.ClusterToken = "cluster-secret"
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("ValidateConfig rejected an authenticated external runtime: %v", err)
	}
}

func TestValidateConfigRejectsInvalidRuntimeUBatchSize(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.RuntimeUBatchSize = -1

	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("ValidateConfig accepted a negative runtime micro-batch size")
	}
}

func TestValidateConfigDoesNotRequireGGUFForCustomEngine(t *testing.T) {
	cfg := NormalizeConfig(DefaultConfigValue())
	cfg.NodeName = "node-a"
	cfg.DataDir = t.TempDir()
	cfg.Engine = cluster.Engine("custom")
	cfg.ModelPath = ""

	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("custom engine was coupled to llama.cpp model path: %v", err)
	}
}
