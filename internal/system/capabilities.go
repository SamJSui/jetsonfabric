package system

import (
	"runtime"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const (
	capabilityEngines           = "engines"
	capabilityContainerRuntimes = "container_runtimes"
	capabilityTegrastats        = "tegrastats"
)

const (
	commandDocker              = "docker"
	commandJetsonFabricRuntime = "jetsonfabric-runtime-worker"
	commandLlamaCLI            = "llama-cli"
	commandLlamaServer         = "llama-server"
	commandNVCC                = "nvcc"
	commandTegrastats          = "tegrastats"
)

func deviceClass() cluster.DeviceClass {
	switch {
	case commandExists(commandTegrastats):
		return cluster.DeviceClassJetson
	case runtime.GOOS == "darwin":
		return cluster.DeviceClassMac
	case runtime.GOOS == "linux":
		return cluster.DeviceClassLinuxPC
	default:
		return cluster.DeviceClassUnknown
	}
}

func engines() []string {
	found := []string{}

	if commandExists(commandLlamaCLI) || commandExists(commandLlamaServer) {
		found = append(found, string(cluster.EngineLlamaCPP))
	}

	if commandExists(commandJetsonFabricRuntime) {
		found = append(found, string(cluster.EngineJetsonFabric))
	}

	return found
}

func containerRuntimes() []string {
	if commandExists(commandDocker) {
		return []string{commandDocker}
	}
	return nil
}

func computeBackends() []string {
	found := []string{string(cluster.ComputeBackendCPU)}
	if cudaAvailable() {
		found = append(found, string(cluster.ComputeBackendCUDA))
	}
	return found
}

func cudaAvailable() bool {
	if commandExists(commandNVCC) {
		return true
	}
	if pathExists("/usr/local/cuda") {
		return true
	}

	entries, err := readDir("/usr/local")
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry, "cuda-") {
			return true
		}
	}
	return false
}
