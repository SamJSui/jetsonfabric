package node

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestModelArtifactSHA256(t *testing.T) {
	content := []byte("GGUF-test-artifact")
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := modelArtifactSHA256(path)
	if err != nil {
		t.Fatalf("hash model: %v", err)
	}
	wantBytes := sha256.Sum256(content)
	want := hex.EncodeToString(wantBytes[:])
	if got != want {
		t.Fatalf("sha256=%q want=%q", got, want)
	}
}
