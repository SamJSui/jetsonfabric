package main

import (
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/agent"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/config"
)

const (
	emptyString  = ""
	csvSeparator = ","
)

func main() {
	controlURL := flag.String("control-url", config.DefaultControlURL(), "control-plane URL")
	joinToken := flag.String("join-token", config.DefaultJoinToken, "agent join token")
	nodeID := flag.String("node-id", emptyString, "stable node identifier")
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
	if llamaURL == emptyString {
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
