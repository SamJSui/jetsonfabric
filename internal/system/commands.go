package system

import (
	"os"
	"os/exec"
)

func commandExists(command string) bool {
	_, err := exec.LookPath(command)
	return err == nil
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func readDir(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names, nil
}
