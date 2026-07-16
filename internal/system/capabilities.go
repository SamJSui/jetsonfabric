package system

import (
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

func detectCapabilities(operatingSystem cluster.OperatingSystem) map[string]any {
	return map[string]any{
		cluster.CapabilityMemoryGB:        memoryGB(operatingSystem),
		cluster.CapabilityDeviceClass:     string(deviceClass(operatingSystem)),
		cluster.CapabilityComputeBackends: computeBackends(),
		capabilityEngines:                 engines(),
		capabilityContainerRuntimes:       containerRuntimes(),
		capabilityTegrastats:              commandExists(commandTegrastats),
	}
}

func deviceClass(operatingSystem cluster.OperatingSystem) cluster.DeviceClass {
	switch {
	case commandExists(commandTegrastats):
		return cluster.DeviceClassJetson
	case operatingSystem == cluster.OperatingSystemDarwin:
		return cluster.DeviceClassMac
	case operatingSystem == cluster.OperatingSystemLinux:
		return cluster.DeviceClassLinuxPC
	default:
		return cluster.DeviceClassUnknown
	}
}

func engines() []string {
	// The runtime worker is a host process, not an engine. The current worker
	// binary hosts the llama.cpp adapter, so any of these installed commands
	// advertise the same inference-engine capability exactly once.
	if commandExists(commandLlamaCLI) ||
		commandExists(commandLlamaServer) ||
		commandExists(commandJetsonFabricRuntime) {
		return []string{string(cluster.EngineLlamaCPP)}
	}
	return nil
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
		if len(entry) >= len("cuda-") && entry[:len("cuda-")] == "cuda-" {
			return true
		}
	}
	return false
}
