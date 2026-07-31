package node

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const runtimeHealthTimeout = 45 * time.Second
const maxRuntimeHealthBody = 64 << 10

type runtimeHealth struct {
	Status             string `json:"status"`
	Engine             string `json:"engine"`
	Mode               string `json:"mode"`
	StageTransport     string `json:"stage_transport"`
	ActivationEncoding string `json:"activation_encoding"`
	KVCacheType        string `json:"kv_cache_type"`
	UBatchSize         int    `json:"ubatch_size"`
}

type RuntimeSupervisor struct {
	cmd *exec.Cmd
	URL string
}

func StartRuntimeSupervisor(ctx context.Context, cfg Config) (*RuntimeSupervisor, string, error) {
	if !cfg.RuntimeAuto() {
		err := waitForRuntimeHealth(
			ctx,
			strings.TrimRight(cfg.RuntimeURL, "/")+"/healthz",
			runtimeHealthTimeout,
			cfg,
		)
		return nil, cfg.RuntimeURL, err
	}

	listen, runtimeURL, err := resolveRuntimeListen(cfg.RuntimeListen)
	if err != nil {
		return nil, "", err
	}

	args := runtimeArgs(cfg, listen)
	cmd := exec.CommandContext(ctx, cfg.RuntimeBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, "", fmt.Errorf("start runtime worker: %w", err)
	}

	supervisor := &RuntimeSupervisor{
		cmd: cmd,
		URL: runtimeURL,
	}

	if err := waitForRuntimeHealth(ctx, runtimeURL+"/healthz", runtimeHealthTimeout, cfg); err != nil {
		_ = supervisor.Stop(context.Background())
		return nil, "", err
	}

	return supervisor, runtimeURL, nil
}

func runtimeArgs(cfg Config, listen string) []string {
	args := []string{
		"--listen", listen,
		"--node-name", cfg.NodeName,
		"--engine", string(cfg.Engine),
		"--stage-transport", cfg.RuntimeStageTransport,
		"--activation-encoding", cfg.RuntimeActivationEncoding,
		"--kv-cache-type", cfg.RuntimeKVCacheType,
		"--compute-backend", cfg.RuntimeComputeBackend,
		"--model", cfg.Model,
		"--model-path", cfg.ModelPath,
		"--ctx-size", strconv.Itoa(cfg.RuntimeCtxSize),
		"--ubatch-size", strconv.Itoa(cfg.RuntimeUBatchSize),
		"--n-gpu-layers", strconv.Itoa(cfg.RuntimeNGPULayers),
		"--threads", strconv.Itoa(cfg.RuntimeThreads),
		"--http-workers", strconv.Itoa(cfg.RuntimeHTTPWorkers),
		"--mode", cfg.RuntimeMode,
		"--stage-index", strconv.Itoa(cfg.StageIndex),
		"--stage-count", strconv.Itoa(cfg.StageCount),
		"--layer-start", strconv.Itoa(cfg.LayerStart),
		"--layer-end", strconv.Itoa(cfg.LayerEnd),
	}
	if cfg.RuntimeStartIdle {
		args = append(args, "--idle")
	}
	return args
}

func (s *RuntimeSupervisor) Stop(ctx context.Context) error {
	if s == nil || s.cmd == nil || s.cmd.Process == nil {
		return nil
	}

	_ = s.cmd.Process.Signal(syscall.SIGTERM)

	done := make(chan error, 1)
	go func() {
		done <- s.cmd.Wait()
	}()

	select {
	case <-ctx.Done():
		_ = s.cmd.Process.Kill()
		return ctx.Err()
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
		return <-done
	}
}

func resolveRuntimeListen(configured string) (string, string, error) {
	configured = strings.TrimSpace(configured)
	if configured == "" {
		configured = DefaultRuntimeListen
	}

	host, portText, err := net.SplitHostPort(configured)
	if err != nil {
		return "", "", fmt.Errorf("parse runtime listen address %q: %w", configured, err)
	}

	if strings.TrimSpace(host) == "" {
		host = "127.0.0.1"
	}

	if portText != "" && portText != "0" {
		listen := net.JoinHostPort(host, portText)
		return listen, runtimeURLFromListen(listen), nil
	}

	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return "", "", fmt.Errorf("reserve runtime port: %w", err)
	}
	defer listener.Close()

	tcpAddr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return "", "", fmt.Errorf("runtime listener returned non-TCP address %s", listener.Addr())
	}

	listen := net.JoinHostPort(host, strconv.Itoa(tcpAddr.Port))
	return listen, runtimeURLFromListen(listen), nil
}

func runtimeURLFromListen(listen string) string {
	host, port, err := net.SplitHostPort(listen)
	if err != nil {
		return "http://" + listen
	}

	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}

	return "http://" + net.JoinHostPort(host, port)
}

func waitForRuntimeHealth(
	ctx context.Context,
	healthURL string,
	timeout time.Duration,
	cfg Config,
) error {
	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	var lastErr error

	for {
		req, err := http.NewRequestWithContext(waitCtx, http.MethodGet, healthURL, nil)
		if err != nil {
			return err
		}

		resp, err := client.Do(req)
		if err == nil {
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				health, decodeErr := decodeRuntimeHealth(resp.Body)
				_ = resp.Body.Close()
				if decodeErr != nil {
					return fmt.Errorf("decode runtime health at %s: %w", healthURL, decodeErr)
				}
				if err := validateRuntimeHealth(health, cfg); err != nil {
					return fmt.Errorf("runtime compatibility check at %s: %w", healthURL, err)
				}
				return nil
			}
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("runtime health returned %s", resp.Status)
		} else {
			lastErr = err
		}

		select {
		case <-waitCtx.Done():
			if lastErr != nil {
				return fmt.Errorf("runtime did not become healthy at %s: %w", healthURL, lastErr)
			}
			return fmt.Errorf("runtime did not become healthy at %s", healthURL)
		case <-ticker.C:
		}
	}
}

func decodeRuntimeHealth(reader io.Reader) (runtimeHealth, error) {
	var health runtimeHealth
	decoder := json.NewDecoder(io.LimitReader(reader, maxRuntimeHealthBody))
	if err := decoder.Decode(&health); err != nil {
		return runtimeHealth{}, err
	}
	return health, nil
}

func validateRuntimeHealth(health runtimeHealth, cfg Config) error {
	expected := map[string][2]string{
		"engine":              {health.Engine, string(cfg.Engine)},
		"mode":                {health.Mode, cfg.RuntimeMode},
		"stage_transport":     {health.StageTransport, cfg.RuntimeStageTransport},
		"activation_encoding": {health.ActivationEncoding, cfg.RuntimeActivationEncoding},
		"kv_cache_type":       {health.KVCacheType, cfg.RuntimeKVCacheType},
	}
	for field, values := range expected {
		if values[0] != values[1] {
			return fmt.Errorf("%s=%q, want %q", field, values[0], values[1])
		}
	}
	if health.Status != "ok" {
		return fmt.Errorf("status=%q, want %q", health.Status, "ok")
	}
	if health.UBatchSize != cfg.RuntimeUBatchSize {
		return fmt.Errorf("ubatch_size=%d, want %d", health.UBatchSize, cfg.RuntimeUBatchSize)
	}
	return nil
}
