package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/config"
	"github.com/SamJSui/jetsonfabric/internal/control"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
)

const (
	flagHost       = "host"
	flagPort       = "port"
	flagJoinToken  = "join-token"
	flagModels     = "models"
	flagBenchmarks = "benchmarks"
)

const (
	usageHost       = "host interface to bind"
	usagePort       = "port to bind"
	usageJoinToken  = "agent join token"
	usageModels     = "model registry JSON path"
	usageBenchmarks = "benchmark JSONL output path"
)

const (
	addressFormat       = "%s:%d"
	logLoadRegistry     = "load model registry: %v"
	logControlListening = "JetsonFabric control plane listening on http://%s"
)

func main() {
	host := flag.String(flagHost, config.DefaultControlHost, usageHost)
	port := flag.Int(flagPort, config.DefaultControlPort, usagePort)
	joinToken := flag.String(flagJoinToken, config.DefaultJoinToken, usageJoinToken)
	modelsPath := flag.String(flagModels, config.DefaultModelRegistryPath(), usageModels)
	benchmarksPath := flag.String(flagBenchmarks, config.DefaultBenchmarksPath(), usageBenchmarks)
	flag.Parse()

	registry, err := modelregistry.Load(*modelsPath)
	if err != nil {
		log.Fatalf(logLoadRegistry, err)
	}

	server := control.NewServer(*joinToken, registry, control.WithBenchmarkRecorder(benchmarks.NewJSONLRecorder(*benchmarksPath)))
	addr := fmt.Sprintf(addressFormat, *host, *port)
	log.Printf(logControlListening, addr)
	if err := http.ListenAndServe(addr, server.Router()); err != nil {
		log.Fatal(err)
	}
}
