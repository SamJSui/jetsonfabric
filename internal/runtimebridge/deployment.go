package runtimebridge

import (
	"net/http"
	"net/url"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

const (
	runtimeDeploymentStatusPath   = "/v1/deployment"
	runtimeDeploymentLoadPath     = "/v1/deployment/load"
	runtimeDeploymentActivatePath = "/v1/deployment/activate"
	runtimeDeploymentUnloadPath   = "/v1/deployment/unload"
)

// DeploymentProxy exposes node-local runtime lifecycle operations without
// making a loopback runtime address reachable from other physical hosts.
type DeploymentProxy struct {
	runtimeURL *url.URL
	client     *http.Client
}

func NewDeploymentProxy(runtimeURL string) (*DeploymentProxy, error) {
	parsed, err := parseRuntimeURL(runtimeURL)
	if err != nil {
		return nil, err
	}
	return &DeploymentProxy{
		runtimeURL: parsed,
		client:     &http.Client{Timeout: 10 * time.Minute},
	}, nil
}

func (p *DeploymentProxy) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	runtimePath, method, ok := deploymentRuntimeTarget(req.URL.Path)
	if !ok {
		writeJSON(w, http.StatusNotFound, errorPayload("runtime_deployment_route_not_found", "unknown runtime deployment route"))
		return
	}
	if req.Method != method {
		writeJSON(w, http.StatusMethodNotAllowed, errorPayload("method_not_allowed", "runtime deployment route uses "+method))
		return
	}

	target := *p.runtimeURL
	target.Path = joinPath(target.Path, runtimePath)
	target.RawQuery = req.URL.RawQuery
	outbound, err := http.NewRequestWithContext(req.Context(), req.Method, target.String(), req.Body)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_request_invalid", err.Error()))
		return
	}
	copyHeaders(outbound.Header, req.Header)
	removeHopByHopHeaders(outbound.Header)
	outbound.ContentLength = req.ContentLength
	outbound.TransferEncoding = append([]string(nil), req.TransferEncoding...)

	resp, err := p.client.Do(outbound)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_deployment_unreachable", err.Error()))
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp)
}

func deploymentRuntimeTarget(path string) (string, string, bool) {
	switch path {
	case api.PathRuntimeDeploymentStatus:
		return runtimeDeploymentStatusPath, http.MethodGet, true
	case api.PathRuntimeDeploymentLoad:
		return runtimeDeploymentLoadPath, http.MethodPost, true
	case api.PathRuntimeDeploymentActivate:
		return runtimeDeploymentActivatePath, http.MethodPost, true
	case api.PathRuntimeDeploymentUnload:
		return runtimeDeploymentUnloadPath, http.MethodPost, true
	default:
		return "", "", false
	}
}
