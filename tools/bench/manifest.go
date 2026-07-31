package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const benchmarkManifestVersion = 1

type BenchmarkManifest struct {
	Version  int                  `json:"version"`
	Name     string               `json:"name"`
	Metadata map[string]any       `json:"metadata,omitempty"`
	Suites   []BenchmarkSuiteSpec `json:"suites"`
}

type BenchmarkSuiteSpec struct {
	Name        string         `json:"name"`
	Endpoints   []string       `json:"endpoints"`
	Request     string         `json:"request"`
	Count       int            `json:"count"`
	Warmup      int            `json:"warmup"`
	Concurrency int            `json:"concurrency"`
	Timeout     string         `json:"timeout,omitempty"`
	Stream      bool           `json:"stream"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type BenchmarkReport struct {
	ManifestPath string         `json:"manifest_path"`
	Name         string         `json:"name"`
	Metadata     map[string]any `json:"metadata,omitempty"`
	Host         BenchmarkHost  `json:"host"`
	StartedAt    time.Time      `json:"started_at"`
	FinishedAt   time.Time      `json:"finished_at"`
	Suites       []Summary      `json:"suites"`
}

type BenchmarkHost struct {
	OS   string `json:"os"`
	Arch string `json:"arch"`
}

func runManifest(
	ctx context.Context,
	client *http.Client,
	path string,
) (BenchmarkReport, error) {
	manifest, absolutePath, err := loadManifest(path)
	if err != nil {
		return BenchmarkReport{}, err
	}
	report := BenchmarkReport{
		ManifestPath: absolutePath,
		Name:         manifest.Name,
		Metadata:     manifest.Metadata,
		Host:         BenchmarkHost{OS: runtime.GOOS, Arch: runtime.GOARCH},
		StartedAt:    time.Now().UTC(),
		Suites:       make([]Summary, 0, len(manifest.Suites)),
	}
	base := filepath.Dir(absolutePath)
	for _, suite := range manifest.Suites {
		requestPath := suite.Request
		if !filepath.IsAbs(requestPath) {
			requestPath = filepath.Join(base, requestPath)
		}
		request, err := loadCompletionRequest(requestPath)
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("suite %q request: %w", suite.Name, err)
		}
		if suite.Stream && !request.Streaming {
			request, err = enableStreaming(request)
			if err != nil {
				return BenchmarkReport{}, fmt.Errorf("suite %q streaming: %w", suite.Name, err)
			}
		}
		timeout := 2 * time.Minute
		if strings.TrimSpace(suite.Timeout) != "" {
			timeout, err = time.ParseDuration(suite.Timeout)
			if err != nil {
				return BenchmarkReport{}, fmt.Errorf("suite %q timeout: %w", suite.Name, err)
			}
		}
		summary, err := runBenchmarkEndpoints(
			ctx,
			client,
			suite.Endpoints,
			request,
			suite.Count,
			suite.Warmup,
			suite.Concurrency,
			timeout,
		)
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("suite %q: %w", suite.Name, err)
		}
		summary.Name = suite.Name
		summary.Metadata = suite.Metadata
		report.Suites = append(report.Suites, summary)
	}
	report.FinishedAt = time.Now().UTC()
	return report, nil
}

func loadManifest(path string) (BenchmarkManifest, string, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return BenchmarkManifest{}, "", err
	}
	file, err := os.Open(absolutePath)
	if err != nil {
		return BenchmarkManifest{}, "", err
	}
	defer file.Close()

	var manifest BenchmarkManifest
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return BenchmarkManifest{}, "", err
	}
	if manifest.Version != benchmarkManifestVersion {
		return BenchmarkManifest{}, "", fmt.Errorf(
			"manifest version=%d, want %d",
			manifest.Version,
			benchmarkManifestVersion,
		)
	}
	if strings.TrimSpace(manifest.Name) == "" {
		return BenchmarkManifest{}, "", fmt.Errorf("manifest name is required")
	}
	if len(manifest.Suites) == 0 {
		return BenchmarkManifest{}, "", fmt.Errorf("manifest requires at least one suite")
	}
	names := make(map[string]struct{}, len(manifest.Suites))
	for index, suite := range manifest.Suites {
		if strings.TrimSpace(suite.Name) == "" {
			return BenchmarkManifest{}, "", fmt.Errorf("suite %d name is required", index)
		}
		if _, exists := names[suite.Name]; exists {
			return BenchmarkManifest{}, "", fmt.Errorf("duplicate suite name %q", suite.Name)
		}
		names[suite.Name] = struct{}{}
		if strings.TrimSpace(suite.Request) == "" {
			return BenchmarkManifest{}, "", fmt.Errorf("suite %q request is required", suite.Name)
		}
		if len(suite.Endpoints) == 0 {
			return BenchmarkManifest{}, "", fmt.Errorf("suite %q requires endpoints", suite.Name)
		}
	}
	return manifest, absolutePath, nil
}
