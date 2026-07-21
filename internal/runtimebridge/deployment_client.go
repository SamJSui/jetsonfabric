package runtimebridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

type DeploymentIdentity struct {
	DeploymentID string `json:"deployment_id"`
	ModelID      string `json:"model_id"`
}

type DeploymentStatus struct {
	Resident   bool                `json:"resident"`
	Active     bool                `json:"active"`
	State      string              `json:"state"`
	Deployment *DeploymentIdentity `json:"deployment"`
}

type DeploymentOperationResponse struct {
	DeploymentStatus
	Loaded    bool `json:"loaded,omitempty"`
	Activated bool `json:"activated,omitempty"`
	Unloaded  bool `json:"unloaded,omitempty"`
}

type LoadDeploymentRequest struct {
	DeploymentID   string `json:"deployment_id"`
	ModelID        string `json:"model_id"`
	Engine         string `json:"engine"`
	ComputeBackend string `json:"compute_backend"`
	ModelPath      string `json:"model_path"`
	CtxSize        int    `json:"ctx_size"`
	NGPULayers     int    `json:"n_gpu_layers"`
	Threads        int    `json:"threads"`
	Mode           string `json:"mode"`
	StageIndex     int    `json:"stage_index"`
	StageCount     int    `json:"stage_count"`
	LayerStart     int    `json:"layer_start"`
	LayerEnd       int    `json:"layer_end"`
}

type DeploymentClient interface {
	Status(context.Context, string) (DeploymentStatus, error)
	Load(context.Context, string, LoadDeploymentRequest) (DeploymentOperationResponse, error)
	Activate(context.Context, string, string) (DeploymentOperationResponse, error)
	Unload(context.Context, string, string) (DeploymentOperationResponse, error)
}

type HTTPDeploymentClient struct {
	client *http.Client
}

func NewHTTPDeploymentClient(timeout time.Duration) *HTTPDeploymentClient {
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &HTTPDeploymentClient{client: &http.Client{Timeout: timeout}}
}

func (c *HTTPDeploymentClient) Status(ctx context.Context, nodeURL string) (DeploymentStatus, error) {
	var response DeploymentStatus
	err := c.do(ctx, http.MethodGet, nodeURL, api.PathRuntimeDeploymentStatus, nil, &response)
	return response, err
}

func (c *HTTPDeploymentClient) Load(ctx context.Context, nodeURL string, request LoadDeploymentRequest) (DeploymentOperationResponse, error) {
	var response DeploymentOperationResponse
	err := c.do(ctx, http.MethodPost, nodeURL, api.PathRuntimeDeploymentLoad, request, &response)
	if err == nil && !response.Loaded {
		err = fmt.Errorf("runtime load response did not confirm loaded=true")
	}
	return response, err
}

func (c *HTTPDeploymentClient) Activate(ctx context.Context, nodeURL string, deploymentID string) (DeploymentOperationResponse, error) {
	var response DeploymentOperationResponse
	err := c.do(ctx, http.MethodPost, nodeURL, api.PathRuntimeDeploymentActivate, map[string]string{"deployment_id": deploymentID}, &response)
	if err == nil && !response.Activated {
		err = fmt.Errorf("runtime activation response did not confirm activated=true")
	}
	return response, err
}

func (c *HTTPDeploymentClient) Unload(ctx context.Context, nodeURL string, deploymentID string) (DeploymentOperationResponse, error) {
	var response DeploymentOperationResponse
	err := c.do(ctx, http.MethodPost, nodeURL, api.PathRuntimeDeploymentUnload, map[string]string{"deployment_id": deploymentID}, &response)
	if err == nil && !response.Unloaded {
		err = fmt.Errorf("runtime unload response did not confirm unloaded=true")
	}
	return response, err
}

func (c *HTTPDeploymentClient) do(ctx context.Context, method string, nodeURL string, path string, body any, output any) error {
	target, err := deploymentTarget(nodeURL, path)
	if err != nil {
		return err
	}
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode runtime deployment request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		return fmt.Errorf("create runtime deployment request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("runtime deployment request %s: %w", target, err)
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("read runtime deployment response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var envelope struct {
			Error   string `json:"error"`
			Message string `json:"message"`
		}
		_ = json.Unmarshal(payload, &envelope)
		message := strings.TrimSpace(envelope.Message)
		if message == "" {
			message = strings.TrimSpace(string(payload))
		}
		return fmt.Errorf("runtime deployment request returned %s: %s: %s", resp.Status, envelope.Error, message)
	}
	if err := json.Unmarshal(payload, output); err != nil {
		return fmt.Errorf("decode runtime deployment response: %w", err)
	}
	return nil
}

func deploymentTarget(nodeURL string, path string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(nodeURL))
	if err != nil {
		return "", fmt.Errorf("parse node URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("node URL must include scheme and host")
	}
	parsed.Path = joinPath(parsed.Path, path)
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}
