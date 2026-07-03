package system

import "github.com/SamJSui/jetsonfabric/internal/cluster"

func loadAverage(operatingSystem cluster.OperatingSystem) []float64 {
	switch operatingSystem {
	case cluster.OperatingSystemLinux:
		return procLoadAverage()
	default:
		return nil
	}
}
