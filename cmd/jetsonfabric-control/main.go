package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/control"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
)

func main() {
	host := flag.String("host", "127.0.0.1", "host interface to bind")
	port := flag.Int("port", 52415, "port to bind")
	joinToken := flag.String("join-token", "dev-token", "agent join token")
	modelsPath := flag.String("models", filepath.Join("configs", "models.example.json"), "model registry JSON path")
	benchmarksPath := flag.String("benchmarks", filepath.Join("data", "benchmarks.jsonl"), "benchmark JSONL output path")
	flag.Parse()

	registry, err := modelregistry.Load(*modelsPath)
	if err != nil {
		log.Fatalf("load model registry: %v", err)
	}

	server := control.NewServer(*joinToken, registry, control.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(*benchmarksPath)))
	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("JetsonFabric control plane listening on http://%s", addr)
	if err := http.ListenAndServe(addr, server.Router()); err != nil {
		log.Fatal(err)
	}
}
