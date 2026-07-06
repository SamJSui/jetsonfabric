package runtimegateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

type StageProxy struct {
	runtimeURL *url.URL
	client     *http.Client
}

func NewStageProxy(runtimeURL string) (*StageProxy, error) {
	parsed, err := parseRuntimeURL(runtimeURL)
	if err != nil {
		return nil, err
	}
	return &StageProxy{
		runtimeURL: parsed,
		client:     &http.Client{Timeout: 2 * time.Minute},
	}, nil
}

func (p *StageProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorPayload("method_not_allowed", "stage execution requires POST"))
		return
	}

	outbound, err := p.newRuntimeRequest(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_request_invalid", err.Error()))
		return
	}

	resp, err := p.client.Do(outbound)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_stage_unreachable", err.Error()))
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp)
}

func parseRuntimeURL(runtimeURL string) (*url.URL, error) {
	runtimeURL = strings.TrimSpace(runtimeURL)
	if runtimeURL == "" {
		return nil, fmt.Errorf("runtime URL is required")
	}
	parsed, err := url.Parse(runtimeURL)
	if err != nil {
		return nil, fmt.Errorf("parse runtime URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("runtime URL must include scheme and host")
	}
	return parsed, nil
}

func (p *StageProxy) newRuntimeRequest(req *http.Request) (*http.Request, error) {
	target := *p.runtimeURL
	target.Path = joinPath(target.Path, api.PathLayerSplitStage)
	target.RawQuery = req.URL.RawQuery

	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, fmt.Errorf("read stage request body: %w", err)
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return nil, fmt.Errorf("stage request body is required")
	}

	outbound, err := http.NewRequestWithContext(
		req.Context(),
		req.Method,
		target.String(),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

	copyHeaders(outbound.Header, req.Header)
	outbound.Header.Del("Transfer-Encoding")
	outbound.Header.Set("Content-Type", "application/json")
	outbound.ContentLength = int64(len(body))

	return outbound, nil
}

func copyResponse(w http.ResponseWriter, resp *http.Response) {
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func joinPath(base string, suffix string) string {
	base = strings.TrimRight(base, "/")
	suffix = "/" + strings.TrimLeft(suffix, "/")
	if base == "" {
		return suffix
	}
	return base + suffix
}

func copyHeaders(dst http.Header, src http.Header) {
	for key, values := range src {
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func errorPayload(code string, message string) map[string]string {
	return map[string]string{"error": code, "message": message}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
