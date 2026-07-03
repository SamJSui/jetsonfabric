package system

import (
	"os/exec"
	"strconv"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

const (
	bytesPerGiB = 1024 * 1024 * 1024
)

func memoryGB(operatingSystem cluster.OperatingSystem) float64 {
	switch operatingSystem {
	case cluster.OperatingSystemLinux:
		return linuxMemoryGB()
	case cluster.OperatingSystemDarwin:
		return darwinMemoryGB()
	default:
		return 0
	}
}

func linuxMemoryGB() float64 {
	return procMemTotalGB()
}

func darwinMemoryGB() float64 {
	output, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0
	}

	bytes, err := strconv.ParseFloat(strings.TrimSpace(string(output)), 64)
	if err != nil {
		return 0
	}

	return round2(bytes / bytesPerGiB)
}
