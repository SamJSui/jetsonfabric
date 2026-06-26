package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/agent"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func main() {
	controlURL := flag.String("control-url", "http://127.0.0.1:52415", "control-plane URL")
	joinToken := flag.String("join-token", "dev-token", "agent join token")
	nodeID := flag.String("node-id", "", "stable node identifier")
	interval := flag.Duration("interval", 10*time.Second, "heartbeat interval")
	once := flag.Bool("once", false, "send one heartbeat and exit")
	llamaURL := flag.String("llama-url", "", "base URL for a llama.cpp OpenAI-compatible server")
	llamaModels := flag.String("llama-models", "", "comma-separated JetsonFabric model IDs served by the llama backend")
	flag.Parse()

	if *nodeID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			log.Fatalf("detect hostname: %v", err)
		}
		*nodeID = hostname
	}

	client := agent.NewClient(*controlURL, *joinToken, *nodeID, advertisedBackends(*llamaURL, *llamaModels))
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

func advertisedBackends(llamaURL string, llamaModels string) []cluster.RuntimeBackend {
	llamaURL = strings.TrimSpace(llamaURL)
	if llamaURL == "" {
		return nil
	}
	return []cluster.RuntimeBackend{
		{
			ID:               cluster.BackendIDLlamaLocal,
			Kind:             cluster.RuntimeKindLlamaCPP,
			BaseURL:          llamaURL,
			Models:           splitCSV(llamaModels),
			OpenAICompatible: true,
		},
	}
}

func splitCSV(value string) []string {
	parts := strings.Split(value, ",")
	items := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			items = append(items, part)
		}
	}
	return items
}
