package node

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestComputeModelArtifactSHA256(t *testing.T) {
	content := []byte("GGUF-test-artifact")
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := computeModelArtifactSHA256(path)
	if err != nil {
		t.Fatalf("hash model: %v", err)
	}
	wantBytes := sha256.Sum256(content)
	want := hex.EncodeToString(wantBytes[:])
	if got != want {
		t.Fatalf("sha256=%q want=%q", got, want)
	}
}

func TestComputeModelArtifactSHA256FromJFMPackage(t *testing.T) {
	wantBytes := sha256.Sum256([]byte("source GGUF"))
	header := make([]byte, 64)
	copy(header[:8], "JFMODEL2")
	binary.LittleEndian.PutUint32(header[8:12], 2)
	copy(header[24:56], wantBytes[:])

	path := t.TempDir()
	if err := os.WriteFile(filepath.Join(path, "manifest.jfm"), header, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := computeModelArtifactSHA256(path)
	if err != nil {
		t.Fatalf("read JFM identity: %v", err)
	}
	if want := hex.EncodeToString(wantBytes[:]); got != want {
		t.Fatalf("sha256=%q want=%q", got, want)
	}
}

func TestComputeModelArtifactSHA256RejectsInvalidJFMPackage(t *testing.T) {
	path := t.TempDir()
	if err := os.WriteFile(filepath.Join(path, "manifest.jfm"), []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := computeModelArtifactSHA256(path); err == nil {
		t.Fatal("invalid JFM package was accepted")
	}
}
