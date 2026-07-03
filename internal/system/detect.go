package system

import (
	"os"
	"runtime"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const (
	metricLoadAverage = "load_average"
	metricQueueDepth  = "queue_depth"
	metricJetsonHint  = "jetson_hint"

	jetsonHintTegrastats = "tegrastats_available"
)

type Snapshot struct {
	Hostname     string         `json:"hostname"`
	Arch         string         `json:"arch"`
	OS           string         `json:"os"`
	Capabilities map[string]any `json:"capabilities"`
	Metrics      map[string]any `json:"metrics"`
}

func Detect() Snapshot {
	hostname, _ := os.Hostname()

	capabilities := map[string]any{
		cluster.CapabilityMemoryGB:        memoryGB(),
		cluster.CapabilityDeviceClass:     string(deviceClass()),
		cluster.CapabilityComputeBackends: computeBackends(),
		capabilityEngines:                 engines(),
		capabilityContainerRuntimes:       containerRuntimes(),
		capabilityTegrastats:              commandExists(commandTegrastats),
	}

	metrics := map[string]any{
		metricLoadAverage: loadAverage(),
		metricQueueDepth:  0,
	}

	if commandExists(commandTegrastats) {
		metrics[metricJetsonHint] = jetsonHintTegrastats
	}

	return Snapshot{
		Hostname:     hostname,
		Arch:         runtime.GOARCH,
		OS:           runtime.GOOS,
		Capabilities: capabilities,
		Metrics:      metrics,
	}
}
