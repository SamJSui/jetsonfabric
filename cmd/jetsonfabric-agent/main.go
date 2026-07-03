package main

import (
	"errors"
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
	emptyString    = ""
	defaultTimeout = 60 * time.Second
)

type agentConfig struct {
	controlURL         string
	joinToken          string
	nodeName           string
	listen             string
	advertiseURL       string
	modelArtifactsPath string
	interval           time.Duration
	once               bool
	llamaURL           string
	model              string
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	cfg, err := parseAgentConfig(args)
	if err != nil {
		return err
	}
	if err := validateAgentConfig(cfg); err != nil {
		return err
	}

	if cfg.llamaURL != emptyString {
		startProxyServer(cfg.listen, cfg.llamaURL, cfg.nodeName)
	}

	if err := logModelArtifacts(cfg.modelArtifactsPath, cfg.model); err != nil {
		return err
	}

	client := agent.NewClient(cfg.controlURL, cfg.joinToken, cfg.nodeName, advertisedBackends(cfg.advertiseURL, cfg.llamaURL, cfg.model))
	for {
		if err := client.SendHeartbeat(); err != nil {
			log.Printf("heartbeat failed: %v", err)
		} else {
			log.Printf("heartbeat sent: %s", cfg.nodeName)
		}
		if cfg.once {
			return nil
		}
		time.Sleep(cfg.interval)
	}
}

func parseAgentConfig(args []string) (agentConfig, error) {
	cfg := agentConfig{}
	fs := flag.NewFlagSet("jetsonfabric-agent", flag.ContinueOnError)
	fs.StringVar(&cfg.controlURL, "control-url", config.DefaultControlURL(), "control-plane URL")
	fs.StringVar(&cfg.joinToken, "join-token", config.DefaultJoinToken, "agent join token")
	fs.StringVar(&cfg.nodeName, "node-name", emptyString, "stable node name; defaults to OS hostname")
	fs.StringVar(&cfg.listen, "listen", config.DefaultAgentListen(), "agent proxy listen address")
	fs.StringVar(&cfg.advertiseURL, "advertise-url", config.DefaultAgentURL(), "agent proxy URL advertised to the control plane")
	fs.StringVar(&cfg.modelArtifactsPath, "model-artifacts", config.DefaultModelArtifactsPath(), "model artifact catalog JSON path")
	fs.DurationVar(&cfg.interval, "interval", config.DefaultHeartbeatInterval, "heartbeat interval")
	fs.BoolVar(&cfg.once, "once", false, "send one heartbeat and exit")
	fs.StringVar(&cfg.llamaURL, "llama-url", emptyString, "base URL for a llama.cpp OpenAI-compatible server")
	fs.StringVar(&cfg.model, "model", emptyString, "JetsonFabric model ID served by the llama backend")
	if err := fs.Parse(args); err != nil {
		return agentConfig{}, err
	}
	return resolveAgentConfigDefaults(normalizeAgentConfig(cfg))
}

func normalizeAgentConfig(cfg agentConfig) agentConfig {
	cfg.controlURL = strings.TrimSpace(cfg.controlURL)
	cfg.joinToken = strings.TrimSpace(cfg.joinToken)
	cfg.nodeName = strings.TrimSpace(cfg.nodeName)
	cfg.listen = strings.TrimSpace(cfg.listen)
	cfg.advertiseURL = strings.TrimSpace(cfg.advertiseURL)
	cfg.modelArtifactsPath = strings.TrimSpace(cfg.modelArtifactsPath)
	cfg.llamaURL = strings.TrimSpace(cfg.llamaURL)
	cfg.model = strings.TrimSpace(cfg.model)
	return cfg
}

func resolveAgentConfigDefaults(cfg agentConfig) (agentConfig, error) {
	if cfg.nodeName != emptyString {
		return cfg, nil
	}
	hostname, err := os.Hostname()
	if err != nil {
		return agentConfig{}, fmt.Errorf("detect hostname: %w", err)
	}
	cfg.nodeName = strings.TrimSpace(hostname)
	return cfg, nil
}

func validateAgentConfig(cfg agentConfig) error {
	if cfg.controlURL == emptyString {
		return fmt.Errorf("--control-url is required")
	}
	if cfg.nodeName == emptyString {
		return fmt.Errorf("node name is required")
	}
	if cfg.interval <= 0 {
		return fmt.Errorf("--interval must be greater than zero")
	}
	if cfg.listen == emptyString {
		return fmt.Errorf("--listen is required")
	}
	runtimeConfigured := cfg.llamaURL != emptyString
	modelConfigured := cfg.model != emptyString
	if cfg.once && runtimeConfigured {
		return fmt.Errorf("--once cannot be used with --llama-url because proxied runtimes require a running agent")
	}
	if runtimeConfigured && !modelConfigured {
		return fmt.Errorf("--model is required when --llama-url is set")
	}
	if modelConfigured && !runtimeConfigured {
		return fmt.Errorf("--llama-url is required when --model is set")
	}
	if runtimeConfigured && cfg.advertiseURL == emptyString {
		return fmt.Errorf("--advertise-url is required when --llama-url is set")
	}
	return nil
}

func startProxyServer(listen string, runtimeURL string, nodeName string) {
	backend, err := runtimeclient.NewOpenAIClient(runtimeURL, defaultTimeout)
	if err != nil {
		log.Fatalf("configure llama runtime backend: %v", err)
	}
	server := &http.Server{
		Addr:              listen,
		Handler:           agent.NewServer(backend, agent.WithNodeName(nodeName)).Router(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("JetsonFabric agent proxy listening on http://%s", listen)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("agent proxy failed: %v", err)
		}
	}()
}

func logModelArtifacts(path string, model string) error {
	modelIDs := advertisedModels(model)
	path = strings.TrimSpace(path)
	if path == emptyString || len(modelIDs) == 0 {
		return nil
	}
	artifacts, err := resolveModelArtifacts(path, modelIDs)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			log.Printf("model artifact catalog %s not found; continuing without artifact metadata", path)
			return nil
		}
		return err
	}
	for _, artifact := range artifacts {
		log.Printf("model artifact %s source=%s local_path=%s", artifact.ModelID, artifact.SourceURL, artifact.LocalPath)
	}
	return nil
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

func advertisedBackends(agentURL string, llamaURL string, model string) []cluster.RuntimeBackend {
	agentURL = strings.TrimSpace(agentURL)
	llamaURL = strings.TrimSpace(llamaURL)
	modelIDs := advertisedModels(model)
	if agentURL == emptyString || llamaURL == emptyString || len(modelIDs) == 0 {
		return nil
	}
	return []cluster.RuntimeBackend{
		{
			ID:               cluster.BackendIDLlamaLocal,
			Kind:             cluster.RuntimeKindLlamaCPP,
			BaseURL:          agentURL,
			Models:           modelIDs,
			OpenAICompatible: true,
		},
	}
}

func advertisedModels(model string) []string {
	model = strings.TrimSpace(model)
	if model == emptyString {
		return nil
	}
	return []string{model}
}
