package runtimebridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

const (
	GenerationContentType = "application/x-ndjson"
	runtimeGenerationPath = "/v1/generate"
	maxGenerationBody     = 4 << 20
)

type GenerationRequest struct {
	RequestID  string              `json:"request_id"`
	SessionID  string              `json:"session_id"`
	ModelID    string              `json:"model_id"`
	Prompt     string              `json:"prompt"`
	MaxTokens  int                 `json:"max_tokens"`
	Deployment *DeploymentIdentity `json:"deployment,omitempty"`
	Stages     []clusterplan.Stage `json:"stages"`
}

type GenerationEvent struct {
	Type             string   `json:"type"`
	Token            *uint32  `json:"token,omitempty"`
	Text             string   `json:"text,omitempty"`
	Index            int      `json:"index,omitempty"`
	FinishReason     string   `json:"finish_reason,omitempty"`
	PromptTokens     int      `json:"prompt_tokens,omitempty"`
	CompletionTokens int      `json:"completion_tokens,omitempty"`
	SampledTokens    []uint32 `json:"sampled_tokens,omitempty"`
	StageCalls       int      `json:"stage_calls,omitempty"`
	RemoteStageCalls int      `json:"remote_stage_calls,omitempty"`
	BytesIn          int64    `json:"bytes_in,omitempty"`
	BytesOut         int64    `json:"bytes_out,omitempty"`
	Code             string   `json:"code,omitempty"`
	Message          string   `json:"message,omitempty"`
}

type GenerationStream struct {
	Body   io.ReadCloser
	Header http.Header
}

type GenerationClient interface {
	Start(context.Context, string, GenerationRequest) (GenerationStream, error)
}

type HTTPGenerationClientConfig struct {
	Timeout           time.Duration
	CoordinatorNodeID string
	ClusterToken      string
}

type HTTPGenerationClient struct {
	client            *http.Client
	coordinatorNodeID string
	clusterToken      string
}

func NewHTTPGenerationClient(cfg HTTPGenerationClientConfig) *HTTPGenerationClient {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Minute
	}
	return &HTTPGenerationClient{
		client:            &http.Client{Timeout: cfg.Timeout},
		coordinatorNodeID: strings.TrimSpace(cfg.CoordinatorNodeID),
		clusterToken:      strings.TrimSpace(cfg.ClusterToken),
	}
}

func (c *HTTPGenerationClient) Start(ctx context.Context, nodeURL string, request GenerationRequest) (GenerationStream, error) {
	target, err := generationTarget(nodeURL)
	if err != nil {
		return GenerationStream{}, err
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return GenerationStream{}, fmt.Errorf("encode runtime generation request: %w", err)
	}
	if len(payload) > maxGenerationBody {
		return GenerationStream{}, fmt.Errorf("runtime generation request exceeds %d bytes", maxGenerationBody)
	}
	outbound, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
	if err != nil {
		return GenerationStream{}, fmt.Errorf("create runtime generation request: %w", err)
	}
	outbound.Header.Set("Content-Type", "application/json")
	outbound.Header.Set("Accept", GenerationContentType)
	outbound.Header.Set(api.HeaderCoordinatorNodeID, c.coordinatorNodeID)
	outbound.Header.Set(api.HeaderClusterToken, c.clusterToken)

	response, err := c.client.Do(outbound)
	if err != nil {
		return GenerationStream{}, fmt.Errorf("runtime generation request %s: %w", target, err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer response.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return GenerationStream{}, fmt.Errorf("runtime generation request returned %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	if !matchesMediaType(response.Header.Get("Content-Type"), GenerationContentType) {
		defer response.Body.Close()
		return GenerationStream{}, fmt.Errorf("runtime generation returned content-type %q", response.Header.Get("Content-Type"))
	}
	return GenerationStream{Body: response.Body, Header: response.Header.Clone()}, nil
}

func matchesMediaType(value string, expected string) bool {
	mediaType, _, err := mime.ParseMediaType(value)
	return err == nil && strings.EqualFold(mediaType, expected)
}

func generationTarget(nodeURL string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(nodeURL))
	if err != nil {
		return "", fmt.Errorf("parse generation node URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("generation node URL must include scheme and host")
	}
	parsed.Path = joinPath(parsed.Path, api.PathRuntimeGeneration)
	parsed.RawQuery = ""
	return parsed.String(), nil
}

type GenerationProxy struct {
	runtimeURL *url.URL
	client     *http.Client
}

func NewGenerationProxy(runtimeURL string) (*GenerationProxy, error) {
	parsed, err := parseRuntimeURL(runtimeURL)
	if err != nil {
		return nil, err
	}
	return &GenerationProxy{
		runtimeURL: parsed,
		client:     &http.Client{Timeout: 30 * time.Minute},
	}, nil
}

func (p *GenerationProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorPayload("method_not_allowed", "runtime generation requires POST"))
		return
	}
	if req.ContentLength <= 0 || req.ContentLength > maxGenerationBody {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_generation_body", "runtime generation requires a bounded non-empty body"))
		return
	}
	target := *p.runtimeURL
	target.Path = joinPath(target.Path, runtimeGenerationPath)
	target.RawQuery = ""
	outbound, err := http.NewRequestWithContext(req.Context(), http.MethodPost, target.String(), req.Body)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_request_invalid", err.Error()))
		return
	}
	copyHeaders(outbound.Header, req.Header)
	removeHopByHopHeaders(outbound.Header)
	outbound.Header.Del(api.HeaderCoordinatorNodeID)
	outbound.Header.Del(api.HeaderClusterToken)
	outbound.ContentLength = req.ContentLength

	response, err := p.client.Do(outbound)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_generation_unreachable", err.Error()))
		return
	}
	defer response.Body.Close()
	copyStreamingResponse(w, response)
}

func copyStreamingResponse(w http.ResponseWriter, response *http.Response) {
	copyHeaders(w.Header(), response.Header)
	removeHopByHopHeaders(w.Header())
	w.WriteHeader(response.StatusCode)
	flusher, _ := w.(http.Flusher)
	buffer := make([]byte, 32<<10)
	for {
		read, readErr := response.Body.Read(buffer)
		if read > 0 {
			if _, writeErr := w.Write(buffer[:read]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if readErr != nil {
			return
		}
	}
}
