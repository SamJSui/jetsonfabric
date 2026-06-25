package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/SamJSui/JetsonMesh/internal/agent"
)

func main() {
	controlURL := flag.String("control-url", "http://127.0.0.1:52415", "control-plane URL")
	joinToken := flag.String("join-token", "dev-token", "agent join token")
	nodeID := flag.String("node-id", "", "stable node identifier")
	interval := flag.Duration("interval", 10*time.Second, "heartbeat interval")
	once := flag.Bool("once", false, "send one heartbeat and exit")
	flag.Parse()

	if *nodeID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			log.Fatalf("detect hostname: %v", err)
		}
		*nodeID = hostname
	}

	client := agent.NewClient(*controlURL, *joinToken, *nodeID)
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
