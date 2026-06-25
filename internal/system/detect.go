package system

import (
	"bufio"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
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
	if commandExists("tegrastats") {
		accelerators = append(accelerators, cluster.AcceleratorJetson, cluster.AcceleratorCUDA)
	}
	capabilities := map[string]any{
		cluster.CapabilityMemoryGB:     memoryGB(),
		cluster.CapabilityAccelerators: accelerators,
		"runtimes":                     runtimes(),
		"tegrastats":                   commandExists("tegrastats"),
	}
	metrics := map[string]any{
		"load_average": loadAverage(),
		"queue_depth":  0,
	}
	if commandExists("tegrastats") {
		metrics["jetson_hint"] = "tegrastats_available"
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
		"docker":    "docker",
		"trtexec":   string(cluster.RuntimeKindTensorRT),
		"llama-cli": string(cluster.RuntimeKindLlamaCPP),
		"ollama":    string(cluster.RuntimeKindOllama),
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
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0
		}
		return round2(kb / 1024 / 1024)
	}
	return 0
}

func loadAverage() []float64 {
	content, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return nil
	}
	fields := strings.Fields(string(content))
	if len(fields) < 3 {
		return nil
	}
	values := make([]float64, 0, 3)
	for _, field := range fields[:3] {
		value, err := strconv.ParseFloat(field, 64)
		if err != nil {
			return nil
		}
		values = append(values, round2(value))
	}
	return values
}

func round2(value float64) float64 {
	return float64(int(value*100+0.5)) / 100
}
