package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/agent"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/config"
	"github.com/SamJSui/jetsonfabric/internal/modelartifacts"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

const (
	emptyString   = ""
	csvSeparator  = ","
	addressFormat = "%s:%d"
)

func main() {
	controlURL := flag.String("control-url", config.DefaultControlURL(), "control-plane URL")
	joinToken := flag.String("join-token", config.DefaultJoinToken, "agent join token")
	nodeID := flag.String("node-id", emptyString, "stable node identifier")
	host := flag.String("host", config.DefaultAgentHost, "agent proxy host interface to bind")
	port := flag.Int("port", config.DefaultAgentPort, "agent proxy port to bind")
	advertiseURL := flag.String("advertise-url", config.DefaultAgentURL(), "agent proxy URL advertised to the control plane")
	modelArtifactsPath := flag.String("model-artifacts", config.DefaultModelArtifactsPath(), "model artifact catalog JSON path")
	interval := flag.Duration("interval", config.DefaultHeartbeatInterval, "heartbeat interval")
	once := flag.Bool("once", false, "send one heartbeat and exit")
	llamaURL := flag.String("llama-url", emptyString, "base URL for a llama.cpp OpenAI-compatible server")
	llamaModels := flag.String("llama-models", emptyString, "comma-separated JetsonFabric model IDs served by the llama backend")
	flag.Parse()

	if *nodeID == emptyString {
		hostname, err := os.Hostname()
		if err != nil {
			log.Fatalf("detect hostname: %v", err)
		}
		*nodeID = hostname
	}

	llamaRuntimeURL := strings.TrimSpace(*llamaURL)
	if *once && llamaRuntimeURL != emptyString {
		log.Fatal("--once cannot be used with --llama-url because proxied runtimes require a running agent")
	}
	if llamaRuntimeURL != emptyString {
		startProxyServer(*host, *port, llamaRuntimeURL, *nodeID)
	}

	servedModels := splitCSV(*llamaModels)
	logModelArtifacts(*modelArtifactsPath, servedModels)

	client := agent.NewClient(*controlURL, *joinToken, *nodeID, advertisedBackends(*advertiseURL, llamaRuntimeURL, servedModels))
	for {
		if err := client.SendHeartbeat(); err != nil {
			log.Printf("heartbeat failed: %v", err)
		} else {
			log.Printf("heartbeat sent: %s", *nodeID)
		}
		if *once {
			return
		}
		time.Sleep(*interval)
	}
}

func startProxyServer(host string, port int, runtimeURL string, nodeID string) {
	backend, err := runtimeclient.NewOpenAIClient(runtimeURL, 60*time.Second)
	if err != nil {
		log.Fatalf("configure llama runtime backend: %v", err)
	}
	addr := fmt.Sprintf(addressFormat, host, port)
	server := &http.Server{
		Addr:              addr,
		Handler:           agent.NewServer(backend, agent.WithNodeID(nodeID)).Router(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("JetsonFabric agent proxy listening on http://%s", addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("agent proxy failed: %v", err)
		}
	}()
}

func logModelArtifacts(path string, modelIDs []string) {
	if strings.TrimSpace(path) == emptyString || len(modelIDs) == 0 {
		return
	}
	artifacts, err := resolveModelArtifacts(path, modelIDs)
	if err != nil {
		log.Fatal(err)
	}
	for _, artifact := range artifacts {
		log.Printf("model artifact %s source=%s local_path=%s", artifact.ModelID, artifact.SourceURL, artifact.LocalPath)
	}
}

func resolveModelArtifacts(path string, modelIDs []string) ([]modelartifacts.Artifact, error) {
	catalog, err := modelartifacts.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load model artifacts: %w", err)
	}
	artifacts := make([]modelartifacts.Artifact, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		if slices.ContainsFunc(artifacts, func(artifact modelartifacts.Artifact) bool {
			return artifact.ModelID == modelID
		}) {
			continue
		}
		artifact, ok := catalog.Find(modelID)
		if !ok {
			return nil, fmt.Errorf("model artifact not found for advertised model %s", modelID)
		}
		artifacts = append(artifacts, artifact)
	}
	return artifacts, nil
}

func advertisedBackends(agentURL string, llamaURL string, llamaModels []string) []cluster.RuntimeBackend {
	agentURL = strings.TrimSpace(agentURL)
	llamaURL = strings.TrimSpace(llamaURL)
	if agentURL == emptyString || llamaURL == emptyString {
		return nil
	}
	return []cluster.RuntimeBackend{
		{
			ID:               cluster.BackendIDLlamaLocal,
			Kind:             cluster.RuntimeKindLlamaCPP,
			BaseURL:          agentURL,
			Models:           llamaModels,
			OpenAICompatible: true,
		},
	}
}

func splitCSV(value string) []string {
	parts := strings.Split(value, csvSeparator)
	items := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != emptyString {
			items = append(items, part)
		}
	}
	return items
}
