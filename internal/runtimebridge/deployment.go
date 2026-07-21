package runtimebridge

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/modelartifacts"
)

const (
	runtimeDeploymentStatusPath   = "/v1/deployment"
	runtimeDeploymentLoadPath     = "/v1/deployment/load"
	runtimeDeploymentActivatePath = "/v1/deployment/activate"
	runtimeDeploymentDrainPath    = "/v1/deployment/drain"
	runtimeDeploymentUnloadPath   = "/v1/deployment/unload"
	maxDeploymentRequestBytes     = 1 << 20
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
	body, contentLength, ok := verifiedDeploymentBody(w, req, runtimePath)
	if !ok {
		return
	}
	outbound, err := http.NewRequestWithContext(req.Context(), req.Method, target.String(), body)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_request_invalid", err.Error()))
		return
	}
	copyHeaders(outbound.Header, req.Header)
	removeHopByHopHeaders(outbound.Header)
	outbound.Header.Del(api.HeaderCoordinatorNodeID)
	outbound.Header.Del(api.HeaderClusterToken)
	outbound.ContentLength = contentLength
	outbound.TransferEncoding = append([]string(nil), req.TransferEncoding...)

	resp, err := p.client.Do(outbound)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, errorPayload("runtime_deployment_unreachable", err.Error()))
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp)
}

func verifiedDeploymentBody(w http.ResponseWriter, req *http.Request, runtimePath string) (io.Reader, int64, bool) {
	if runtimePath != runtimeDeploymentLoadPath {
		return req.Body, req.ContentLength, true
	}
	payload, err := io.ReadAll(io.LimitReader(req.Body, maxDeploymentRequestBytes+1))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_load_request", "read deployment load request: "+err.Error()))
		return nil, 0, false
	}
	if len(payload) > maxDeploymentRequestBytes {
		writeJSON(w, http.StatusRequestEntityTooLarge, errorPayload("invalid_load_request", "deployment load request exceeds 1 MiB"))
		return nil, 0, false
	}
	var request struct {
		Epoch       uint64 `json:"epoch"`
		ModelPath   string `json:"model_path"`
		ModelSHA256 string `json:"model_sha256"`
	}
	if err := json.Unmarshal(payload, &request); err != nil {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_load_request", "request body must be valid JSON"))
		return nil, 0, false
	}
	modelPath := strings.TrimSpace(request.ModelPath)
	modelSHA256 := strings.TrimSpace(request.ModelSHA256)
	if request.Epoch == 0 || modelPath == "" ||
		modelSHA256 != strings.ToLower(modelSHA256) || !validSHA256(modelSHA256) {
		writeJSON(w, http.StatusBadRequest, errorPayload("invalid_load_identity", "epoch, model_path, and a 64-character model_sha256 are required"))
		return nil, 0, false
	}
	actualSHA256, err := modelartifacts.ComputeSHA256(modelPath)
	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, errorPayload("model_artifact_unreadable", err.Error()))
		return nil, 0, false
	}
	if actualSHA256 != modelSHA256 {
		writeJSON(w, http.StatusConflict, errorPayload("model_artifact_mismatch", "model artifact SHA-256 does not match the deployment plan"))
		return nil, 0, false
	}
	return bytes.NewReader(payload), int64(len(payload)), true
}

func validSHA256(value string) bool {
	if len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func deploymentRuntimeTarget(path string) (string, string, bool) {
	switch path {
	case api.PathRuntimeDeploymentStatus:
		return runtimeDeploymentStatusPath, http.MethodGet, true
	case api.PathRuntimeDeploymentLoad:
		return runtimeDeploymentLoadPath, http.MethodPost, true
	case api.PathRuntimeDeploymentActivate:
		return runtimeDeploymentActivatePath, http.MethodPost, true
	case api.PathRuntimeDeploymentDrain:
		return runtimeDeploymentDrainPath, http.MethodPost, true
	case api.PathRuntimeDeploymentUnload:
		return runtimeDeploymentUnloadPath, http.MethodPost, true
	default:
		return "", "", false
	}
}
