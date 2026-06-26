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
	flagControlURL  = "control-url"
	flagJoinToken   = "join-token"
	flagNodeID      = "node-id"
	flagInterval    = "interval"
	flagOnce        = "once"
	flagLlamaURL    = "llama-url"
	flagLlamaModels = "llama-models"
)

const (
	usageControlURL  = "control-plane URL"
	usageJoinToken   = "agent join token"
	usageNodeID      = "stable node identifier"
	usageInterval    = "heartbeat interval"
	usageOnce        = "send one heartbeat and exit"
	usageLlamaURL    = "base URL for a llama.cpp OpenAI-compatible server"
	usageLlamaModels = "comma-separated JetsonFabric model IDs served by the llama backend"
)

const (
	logDetectHostnameFailed = "detect hostname: %v"
	logHeartbeatFailed      = "heartbeat failed: %v"
	logHeartbeatSent        = "heartbeat sent: %s"
	csvSeparator            = ","
)

func main() {
	controlURL := flag.String(flagControlURL, config.DefaultControlURL(), usageControlURL)
	joinToken := flag.String(flagJoinToken, config.DefaultJoinToken, usageJoinToken)
	nodeID := flag.String(flagNodeID, "", usageNodeID)
	interval := flag.Duration(flagInterval, config.DefaultHeartbeatInterval, usageInterval)
	once := flag.Bool(flagOnce, false, usageOnce)
	llamaURL := flag.String(flagLlamaURL, "", usageLlamaURL)
	llamaModels := flag.String(flagLlamaModels, "", usageLlamaModels)
	flag.Parse()

	if *nodeID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			log.Fatalf(logDetectHostnameFailed, err)
		}
		*nodeID = hostname
	}

	client := agent.NewClient(*controlURL, *joinToken, *nodeID, advertisedBackends(*llamaURL, *llamaModels))
	for {
		if err := client.SendHeartbeat(); err != nil {
			log.Printf(logHeartbeatFailed, err)
		} else {
			log.Printf(logHeartbeatSent, *nodeID)
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
	parts := strings.Split(value, csvSeparator)
	items := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			items = append(items, part)
		}
	}
	return items
}
