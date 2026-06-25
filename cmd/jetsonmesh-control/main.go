package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"path/filepath"

	"github.com/SamJSui/JetsonMesh/internal/control"
	"github.com/SamJSui/JetsonMesh/internal/modelregistry"
)

func main() {
	host := flag.String("host", "127.0.0.1", "host interface to bind")
	port := flag.Int("port", 52415, "port to bind")
	joinToken := flag.String("join-token", "dev-token", "agent join token")
	modelsPath := flag.String("models", filepath.Join("configs", "models.example.json"), "model registry JSON path")
	flag.Parse()

	registry, err := modelregistry.Load(*modelsPath)
	if err != nil {
		log.Fatalf("load model registry: %v", err)
	}

	server := control.NewServer(*joinToken, registry)
	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("JetsonMesh control plane listening on http://%s", addr)
	if err := http.ListenAndServe(addr, server.Router()); err != nil {
		log.Fatal(err)
	}
}
