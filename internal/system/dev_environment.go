package system

import (
	"os"
	"runtime"
	"strings"
)

func detectWSL() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	return envSet("WSL_DISTRO_NAME") || envSet("WSL_INTEROP")
}

func envSet(name string) bool {
	return strings.TrimSpace(os.Getenv(name)) != ""
}
