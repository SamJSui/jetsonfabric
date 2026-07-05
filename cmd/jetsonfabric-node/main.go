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

type flagValues struct {
	seeds          string
	discoveryModes string
	engine         string
	role           string
}

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
	values := defaultFlagValues(cfg)
	fs := flag.NewFlagSet("jetsonfabric-node", flag.ContinueOnError)
	bindFlags(fs, &cfg, &values)
	if err := fs.Parse(args); err != nil {
		return node.Config{}, err
	}
	return normalizeParsedConfig(cfg, values)
}

func defaultFlagValues(cfg node.Config) flagValues {
	return flagValues{
		discoveryModes: strings.Join(cfg.DiscoveryModes, ","),
		engine:         string(cfg.Engine),
		role:           string(cfg.Role),
	}
}

func bindFlags(fs *flag.FlagSet, cfg *node.Config, values *flagValues) {
	bindCoreFlags(fs, cfg, values)
	bindRoleFlags(fs, cfg, values)
	bindDiscoveryFlags(fs, cfg, values)
	fs.StringVar(&cfg.ModelsPath, "models", cfg.ModelsPath, "model registry JSON path")
	fs.StringVar(&cfg.BenchmarksPath, "benchmarks", cfg.BenchmarksPath, "benchmark JSONL output path")
}

func bindCoreFlags(fs *flag.FlagSet, cfg *node.Config, values *flagValues) {
	fs.StringVar(&cfg.ClusterID, "cluster-id", cfg.ClusterID, "cluster id used to isolate discovered peers")
	fs.StringVar(&cfg.NodeName, "node-name", cfg.NodeName, "stable node name; defaults to OS hostname")
	fs.StringVar(&cfg.Listen, "listen", cfg.Listen, "node facade listen address")
	fs.StringVar(&cfg.APIURL, "advertise-url", cfg.APIURL, "URL this node advertises to peers")
	fs.StringVar(&cfg.DataDir, "data-dir", cfg.DataDir, "directory for stable node identity")
	fs.StringVar(&cfg.RuntimeURL, "runtime-url", cfg.RuntimeURL, "local C++ runtime URL")
	fs.StringVar(&values.engine, "engine", values.engine, "local runtime engine kind")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "model id served by the local runtime")
}

func bindRoleFlags(fs *flag.FlagSet, cfg *node.Config, values *flagValues) {
	fs.StringVar(&values.role, "role", values.role, "node role: auto, jetson, coordinator, worker, or test")
	fs.IntVar(&cfg.LeaderPreference, "leader-preference", cfg.LeaderPreference, "advanced tie-break weight within the same role")
}

func bindDiscoveryFlags(fs *flag.FlagSet, cfg *node.Config, values *flagValues) {
	fs.StringVar(&values.seeds, "seeds", "", "comma-separated peer node API URLs for static discovery")
	fs.StringVar(&values.discoveryModes, "discovery", values.discoveryModes, "comma-separated discovery modes: static,mdns,none")
	fs.DurationVar(&cfg.DiscoveryInterval, "discovery-interval", cfg.DiscoveryInterval, "peer discovery interval")
	fs.DurationVar(&cfg.StaleAfter, "stale-after", cfg.StaleAfter, "member staleness timeout")
	fs.StringVar(&cfg.MDNSService, "mdns-service", cfg.MDNSService, "mDNS service name")
	fs.StringVar(&cfg.MDNSDomain, "mdns-domain", cfg.MDNSDomain, "mDNS domain")
	fs.DurationVar(&cfg.MDNSBrowseTimeout, "mdns-browse-timeout", cfg.MDNSBrowseTimeout, "mDNS browse timeout")
}

func normalizeParsedConfig(cfg node.Config, values flagValues) (node.Config, error) {
	cfg.Engine = cluster.Engine(strings.TrimSpace(values.engine))
	cfg.Role = membership.NodeRole(strings.TrimSpace(values.role))
	cfg.Seeds = splitCSV(values.seeds)
	cfg.DiscoveryModes = splitCSV(values.discoveryModes)
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
