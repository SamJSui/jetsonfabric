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
	bindRuntimeFlags(fs, cfg)
	bindRoleFlags(fs, cfg, values)
	bindDiscoveryFlags(fs, cfg, values)

	fs.StringVar(&cfg.ModelsPath, "models", cfg.ModelsPath, "model registry JSON path")
	fs.StringVar(&cfg.BenchmarksPath, "benchmarks", cfg.BenchmarksPath, "benchmark JSONL output path")
}

func bindCoreFlags(fs *flag.FlagSet, cfg *node.Config, values *flagValues) {
	fs.StringVar(&cfg.ClusterID, "cluster-id", cfg.ClusterID, "cluster id used to isolate discovered peers")
	fs.StringVar(&cfg.NodeName, "node-name", cfg.NodeName, "logical node name; defaults to auto-generated <hostname>-<suffix>")
	fs.StringVar(&cfg.Listen, "listen", cfg.Listen, "node facade listen address; use 0.0.0.0:0 for auto port")
	fs.StringVar(&cfg.APIURL, "advertise-url", cfg.APIURL, "URL this node advertises to peers; defaults to derived URL after bind")
	fs.StringVar(&cfg.DataDir, "data-dir", cfg.DataDir, "directory for stable logical node identity; defaults to .cache/jetsonfabric/nodes/<node-name>")

	fs.StringVar(&values.engine, "engine", values.engine, "local runtime engine kind")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "model id served by the configured startup deployment")
	fs.StringVar(&cfg.ModelPath, "model-path", cfg.ModelPath, "GGUF model path used by the configured startup deployment")
}

func bindRuntimeFlags(fs *flag.FlagSet, cfg *node.Config) {
	fs.StringVar(&cfg.RuntimeURL, "runtime-url", cfg.RuntimeURL, "local runtime URL, or auto to let this node supervise one")
	fs.StringVar(&cfg.RuntimeBin, "runtime-bin", cfg.RuntimeBin, "runtime worker binary path")
	fs.StringVar(&cfg.RuntimeListen, "runtime-listen", cfg.RuntimeListen, "runtime listen address; use 127.0.0.1:0 for auto port")
	fs.StringVar(&cfg.RuntimeComputeBackend, "runtime-compute-backend", cfg.RuntimeComputeBackend, "runtime compute backend: cuda or cpu")
	fs.StringVar(&cfg.RuntimeMode, "runtime-mode", cfg.RuntimeMode, "runtime execution mode")
	fs.IntVar(&cfg.RuntimeCtxSize, "runtime-ctx-size", cfg.RuntimeCtxSize, "runtime context size")
	fs.IntVar(&cfg.RuntimeNGPULayers, "runtime-n-gpu-layers", cfg.RuntimeNGPULayers, "runtime GPU layer count")
	fs.IntVar(&cfg.RuntimeThreads, "runtime-threads", cfg.RuntimeThreads, "runtime CPU thread count")
	fs.BoolVar(&cfg.RuntimeStartIdle, "runtime-idle", cfg.RuntimeStartIdle, "start or describe the local runtime without a resident deployment")
	fs.StringVar(&cfg.RuntimeRevision, "runtime-revision", cfg.RuntimeRevision, "JetsonFabric runtime compatibility revision")
	fs.StringVar(&cfg.RuntimeLlamaCPPRevision, "runtime-llama-cpp-revision", cfg.RuntimeLlamaCPPRevision, "pinned llama.cpp compatibility revision")
	fs.BoolVar(&cfg.RuntimeCUDAActive, "runtime-cuda-active", cfg.RuntimeCUDAActive, "attest that CUDA execution is active for CUDA deployment placement")

	fs.IntVar(&cfg.StageIndex, "stage-index", cfg.StageIndex, "configured startup runtime stage index")
	fs.IntVar(&cfg.StageCount, "stage-count", cfg.StageCount, "configured startup runtime stage count")
	fs.IntVar(&cfg.LayerStart, "layer-start", cfg.LayerStart, "configured startup first layer")
	fs.IntVar(&cfg.LayerEnd, "layer-end", cfg.LayerEnd, "configured startup exclusive layer end")
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

	// Do not call node.ValidateConfig here.
	// Validation must happen inside node.New after:
	//   - auto logical node name
	//   - auto data dir
	//   - runtime defaults
	//   - eventually bound advertise URL
	cfg = node.NormalizeConfig(cfg)

	return cfg, nil
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
