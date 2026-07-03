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
	Hostname     string                  `json:"hostname"`
	Arch         string                  `json:"arch"`
	OS           cluster.OperatingSystem `json:"os"`
	Capabilities map[string]any          `json:"capabilities"`
	Metrics      map[string]any          `json:"metrics"`
}

func Detect() Snapshot {
	hostname, _ := os.Hostname()
	operatingSystem := detectOperatingSystem()

	return Snapshot{
		Hostname:     hostname,
		Arch:         runtime.GOARCH,
		OS:           operatingSystem,
		Capabilities: detectCapabilities(operatingSystem),
		Metrics:      detectMetrics(operatingSystem),
	}
}

func detectMetrics(operatingSystem cluster.OperatingSystem) map[string]any {
	metrics := map[string]any{
		metricLoadAverage: loadAverage(operatingSystem),
		metricQueueDepth:  0,
	}

	if commandExists(commandTegrastats) {
		metrics[metricJetsonHint] = jetsonHintTegrastats
	}

	return metrics
}
