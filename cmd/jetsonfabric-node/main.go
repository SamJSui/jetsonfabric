package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/node"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	cfg, err := parseConfig(args)
	if err != nil {
		return err
	}
	app, err := node.New(cfg)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return app.Run(ctx)
}

func parseConfig(args []string) (node.Config, error) {
	cfg := node.DefaultConfigValue()
	var seeds string
	var discoveryModes string
	var engine string
	var role string

	fs := flag.NewFlagSet("jetsonfabric-node", flag.ContinueOnError)
	fs.StringVar(&cfg.ClusterID, "cluster-id", cfg.ClusterID, "cluster id used to isolate discovered peers")
	fs.StringVar(&cfg.NodeName, "node-name", cfg.NodeName, "stable node name; defaults to OS hostname")
	fs.StringVar(&cfg.Listen, "listen", cfg.Listen, "node facade listen address")
	fs.StringVar(&cfg.APIURL, "advertise-url", cfg.APIURL, "URL this node advertises to peers")
	fs.StringVar(&cfg.DataDir, "data-dir", cfg.DataDir, "directory for stable node identity and state")
	fs.StringVar(&cfg.RuntimeURL, "runtime-url", cfg.RuntimeURL, "local C++ runtime URL")
	fs.StringVar(&engine, "engine", string(cfg.Engine), "local runtime engine kind")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "model id served by the local runtime, when available")
	fs.StringVar(&role, "role", string(cfg.Role), "node role: auto, jetson, coordinator, worker, or test")
	fs.IntVar(&cfg.LeaderPreference, "leader-preference", cfg.LeaderPreference, "advanced tie-break weight within the same role")
	fs.BoolVar(&cfg.ControlEligible, "control-eligible", cfg.ControlEligible, "deprecated; role now derives this")
	fs.IntVar(&cfg.ControlPriority, "control-priority", cfg.ControlPriority, "deprecated alias for leader-preference")
	fs.StringVar(&seeds, "seeds", "", "comma-separated peer node API URLs for static discovery")
	fs.StringVar(&discoveryModes, "discovery", strings.Join(cfg.DiscoveryModes, ","), "comma-separated discovery modes: static,mdns,none")
	fs.DurationVar(&cfg.DiscoveryInterval, "discovery-interval", cfg.DiscoveryInterval, "peer discovery interval")
	fs.DurationVar(&cfg.StaleAfter, "stale-after", cfg.StaleAfter, "member staleness timeout")
	fs.StringVar(&cfg.MDNSService, "mdns-service", cfg.MDNSService, "mDNS service name")
	fs.StringVar(&cfg.MDNSDomain, "mdns-domain", cfg.MDNSDomain, "mDNS domain")
	fs.DurationVar(&cfg.MDNSBrowseTimeout, "mdns-browse-timeout", cfg.MDNSBrowseTimeout, "mDNS browse timeout per discovery tick")
	fs.StringVar(&cfg.JoinToken, "join-token", cfg.JoinToken, "internal join token used by embedded coordinator")
	fs.StringVar(&cfg.ModelsPath, "models", cfg.ModelsPath, "model registry JSON path")
	fs.StringVar(&cfg.BenchmarksPath, "benchmarks", cfg.BenchmarksPath, "benchmark JSONL output path")
	if err := fs.Parse(args); err != nil {
		return node.Config{}, err
	}

	cfg.Engine = cluster.Engine(strings.TrimSpace(engine))
	cfg.Role = membership.NodeRole(strings.TrimSpace(role))
	cfg.Seeds = splitCSV(seeds)
	cfg.DiscoveryModes = splitCSV(discoveryModes)
	cfg = node.NormalizeConfig(cfg)
	return cfg, node.ValidateConfig(cfg)
}

func splitCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}
