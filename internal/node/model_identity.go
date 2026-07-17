package node

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"
)

// modelArtifactSHA256 computes the identity used to prove that every pipeline
// stage loaded the same model artifact. It streams the file instead of reading a
// potentially multi-gigabyte GGUF into memory.
func modelArtifactSHA256(path string) (string, error) {
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
