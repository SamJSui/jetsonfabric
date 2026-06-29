package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/config"
	"github.com/SamJSui/jetsonfabric/internal/control"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
)

type controlConfig struct {
	listen         string
	joinToken      string
	modelsPath     string
	benchmarksPath string
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	cfg, err := parseControlConfig(args)
	if err != nil {
		return err
	}
	if err := validateControlConfig(cfg); err != nil {
		return err
	}

	registry, err := modelregistry.Load(cfg.modelsPath)
	if err != nil {
		return fmt.Errorf("load model registry: %w", err)
	}

	server := control.NewServer(cfg.joinToken, registry, control.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(cfg.benchmarksPath)))
	log.Printf("JetsonFabric control plane listening on http://%s", cfg.listen)
	return http.ListenAndServe(cfg.listen, server.Router())
}

func parseControlConfig(args []string) (controlConfig, error) {
	cfg := controlConfig{}
	fs := flag.NewFlagSet("jetsonfabric-control", flag.ContinueOnError)
	fs.StringVar(&cfg.listen, "listen", config.DefaultControlListen(), "control-plane listen address")
	fs.StringVar(&cfg.joinToken, "join-token", config.DefaultJoinToken, "agent join token")
	fs.StringVar(&cfg.modelsPath, "models", config.DefaultModelRegistryPath(), "model registry JSON path")
	fs.StringVar(&cfg.benchmarksPath, "benchmarks", config.DefaultBenchmarksPath(), "benchmark JSONL output path")
	if err := fs.Parse(args); err != nil {
		return controlConfig{}, err
	}
	return normalizeControlConfig(cfg), nil
}

func normalizeControlConfig(cfg controlConfig) controlConfig {
	cfg.listen = strings.TrimSpace(cfg.listen)
	cfg.joinToken = strings.TrimSpace(cfg.joinToken)
	cfg.modelsPath = strings.TrimSpace(cfg.modelsPath)
	cfg.benchmarksPath = strings.TrimSpace(cfg.benchmarksPath)
	return cfg
}

func validateControlConfig(cfg controlConfig) error {
	if cfg.listen == "" {
		return fmt.Errorf("--listen is required")
	}
	if cfg.modelsPath == "" {
		return fmt.Errorf("--models is required")
	}
	if cfg.benchmarksPath == "" {
		return fmt.Errorf("--benchmarks is required")
	}
	return nil
}
