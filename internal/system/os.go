package system

import (
	"runtime"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func detectOperatingSystem() cluster.OperatingSystem {
	switch runtime.GOOS {
	case "linux":
		return cluster.OperatingSystemLinux
	case "darwin":
		return cluster.OperatingSystemDarwin
	case "windows":
		return cluster.OperatingSystemWindows
	default:
		return cluster.OperatingSystemUnknown
	}
}
