package system

import (
	"bufio"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const (
	procMeminfoPath = "/proc/meminfo"
	procLoadavgPath = "/proc/loadavg"

	meminfoTotalField     = "MemTotal:"
	meminfoValueField     = 1
	loadAverageFieldCount = 3
	kibPerGiB             = 1024 * 1024
)

const (
	capabilityRuntimes   = "runtimes"
	capabilityTegrastats = "tegrastats"
	metricLoadAverage    = "load_average"
	metricQueueDepth     = "queue_depth"
	metricJetsonHint     = "jetson_hint"
	jetsonHintTegrastats = "tegrastats_available"
)

const (
	commandDocker     = "docker"
	commandTrtexec    = "trtexec"
	commandLlamaCLI   = "llama-cli"
	commandOllama     = "ollama"
	commandTegrastats = "tegrastats"
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
	accelerators := []string{}
	if commandExists(commandTegrastats) {
		accelerators = append(accelerators, cluster.AcceleratorJetson, cluster.AcceleratorCUDA)
	}
	capabilities := map[string]any{
		cluster.CapabilityMemoryGB:     memoryGB(),
		cluster.CapabilityAccelerators: accelerators,
		capabilityRuntimes:             runtimes(),
		capabilityTegrastats:           commandExists(commandTegrastats),
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

func runtimes() []string {
	checks := map[string]string{
		commandDocker:   commandDocker,
		commandTrtexec:  string(cluster.RuntimeKindTensorRT),
		commandLlamaCLI: string(cluster.RuntimeKindLlamaCPP),
		commandOllama:   string(cluster.RuntimeKindOllama),
	}
	found := []string{}
	for command, name := range checks {
		if commandExists(command) {
			found = append(found, name)
		}
	}
	return found
}

func commandExists(command string) bool {
	_, err := exec.LookPath(command)
	return err == nil
}

func memoryGB() float64 {
	file, err := os.Open(procMeminfoPath)
	if err != nil {
		return 0
	}
	defer file.Close()
	return parseMemTotalGB(file)
}

func parseMemTotalGB(reader io.Reader) float64 {
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, meminfoTotalField) {
			continue
		}
		fields := strings.Fields(line)
		rawKB, ok := fieldAt(fields, meminfoValueField)
		if !ok {
			return 0
		}
		kb, err := strconv.ParseFloat(rawKB, 64)
		if err != nil {
			return 0
		}
		return round2(kb / kibPerGiB)
	}
	if err := scanner.Err(); err != nil {
		return 0
	}
	return 0
}

func loadAverage() []float64 {
	content, err := os.ReadFile(procLoadavgPath)
	if err != nil {
		return nil
	}
	values := make([]float64, 0, loadAverageFieldCount)
	for _, field := range strings.Fields(string(content)) {
		if len(values) == loadAverageFieldCount {
			break
		}
		value, err := strconv.ParseFloat(field, 64)
		if err != nil {
			return nil
		}
		values = append(values, round2(value))
	}
	if len(values) != loadAverageFieldCount {
		return nil
	}
	return values
}

func round2(value float64) float64 {
	return float64(int(value*100+0.5)) / 100
}

func fieldAt(fields []string, target int) (string, bool) {
	for index, field := range fields {
		if index == target {
			return field, true
		}
	}
	return "", false
}
