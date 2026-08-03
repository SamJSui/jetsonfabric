package modelartifacts

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	jfmManifestHeaderSize = 64
	jfmSourceSHAOffset    = 24
	jfmFormatVersion      = 2
)

var jfmManifestMagic = [8]byte{'J', 'F', 'M', 'O', 'D', 'E', 'L', '2'}

// ComputeSHA256 returns the source-model identity represented by an artifact.
// GGUF files are streamed; JFM packages carry the source GGUF digest in their
// validated manifest header.
func ComputeSHA256(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", fmt.Errorf("model path is required to compute artifact identity")
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("stat model artifact: %w", err)
	}
	if info.IsDir() {
		return computeJFMPackageSHA256(path)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("model artifact must be a regular file or JFM package directory")
	}
	return computeFileSHA256(path)
}

func computeFileSHA256(path string) (string, error) {
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

func computeJFMPackageSHA256(path string) (string, error) {
	manifestPath := filepath.Join(path, "manifest.jfm")
	manifest, err := os.Open(manifestPath)
	if err != nil {
		return "", fmt.Errorf("open JFM manifest: %w", err)
	}
	defer manifest.Close()

	header := make([]byte, jfmManifestHeaderSize)
	if _, err := io.ReadFull(manifest, header); err != nil {
		return "", fmt.Errorf("read JFM manifest header: %w", err)
	}
	if string(header[:len(jfmManifestMagic)]) != string(jfmManifestMagic[:]) ||
		binary.LittleEndian.Uint32(header[8:12]) != jfmFormatVersion ||
		binary.LittleEndian.Uint32(header[60:64]) != 0 {
		return "", fmt.Errorf("JFM manifest has an unsupported header")
	}
	digest := header[jfmSourceSHAOffset : jfmSourceSHAOffset+sha256.Size]
	return hex.EncodeToString(digest), nil
}
