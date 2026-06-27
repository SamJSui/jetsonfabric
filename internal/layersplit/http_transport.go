package layersplit

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

const maxStageErrorBodyBytes = 4096

type HTTPTransport struct {
	client *http.Client
}

func NewHTTPTransport(timeout time.Duration) *HTTPTransport {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &HTTPTransport{
		client: &http.Client{Timeout: timeout},
	}
}

func (t *HTTPTransport) RunStage(ctx context.Context, target StageTarget, req ActivationRequest) (ActivationResponse, error) {
	if target.BaseURL == "" {
		return ActivationResponse{}, fmt.Errorf("stage target %q is missing a base URL", target.NodeID)
	}
	url, err := api.JoinBasePath(target.BaseURL, api.PathLayerSplitStage)
	if err != nil {
		return ActivationResponse{}, err
	}
	body, err := json.Marshal(req)
	if err != nil {
		return ActivationResponse{}, fmt.Errorf("encode stage request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return ActivationResponse{}, fmt.Errorf("create stage request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := t.client.Do(httpReq)
	elapsed := time.Since(start)
	if err != nil {
		return ActivationResponse{}, fmt.Errorf("call stage %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, maxStageErrorBodyBytes))
		return ActivationResponse{}, fmt.Errorf("stage %s returned %s: %s", url, resp.Status, bytes.TrimSpace(snippet))
	}

	var decoded ActivationResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return ActivationResponse{}, fmt.Errorf("decode stage response: %w", err)
	}
	if decoded.Payload == "" {
		return ActivationResponse{}, fmt.Errorf("stage %s returned an empty payload", url)
	}
	return NormalizeResponse(req, decoded, elapsed), nil
}
