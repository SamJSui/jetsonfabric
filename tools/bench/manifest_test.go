package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadManifestValidatesVersionAndResolvesPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	content := `{
		"version": 1,
		"name": "smoke",
		"suites": [{
			"name": "single",
			"endpoints": ["http://node-a.test/v1/chat/completions"],
			"request": "request.json",
			"count": 2,
			"warmup": 1,
			"concurrency": 1,
			"stream": true
		}]
	}`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, absolutePath, err := loadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(absolutePath) || manifest.Name != "smoke" ||
		len(manifest.Suites) != 1 || manifest.Suites[0].Warmup != 1 {
		t.Fatalf("unexpected manifest: %+v path=%q", manifest, absolutePath)
	}
}

func TestLoadManifestRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	content := `{
		"version": 1,
		"name": "smoke",
		"unexpected": true,
		"suites": [{
			"name": "single",
			"endpoints": ["http://node-a.test/v1/chat/completions"],
			"request": "request.json",
			"count": 1,
			"warmup": 0,
			"concurrency": 1,
			"stream": true
		}]
	}`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, err := loadManifest(path)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field error, got %v", err)
	}
}
