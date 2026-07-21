package modelartifacts

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"
)

// ComputeSHA256 streams an artifact so multi-gigabyte GGUF files are not read
// into memory solely to establish deployment identity.
func ComputeSHA256(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("model path is required to compute artifact identity")
	}
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open model artifact: %w", err)
	}
	defer file.Close()

	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", fmt.Errorf("hash model artifact: %w", err)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
