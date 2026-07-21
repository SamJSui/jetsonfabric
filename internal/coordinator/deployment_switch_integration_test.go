package coordinator

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestCoordinatorDeploymentSwitchAdmissionBarrierAndFailureIsolation(t *testing.T) {
	inferenceStarted := make(chan struct{})
	releaseInference := make(chan struct{})
	var inferenceOnce sync.Once
	stage := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.ModelID == "model-a" {
			inferenceOnce.Do(func() { close(inferenceStarted) })
			<-releaseInference
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, 17)
		metadata := responseMetadataForCoordinator(req, stagewire.PayloadKindSampledToken)
		metadata.Message = "token"
		metadata.CompletionTokens = 1
		return stagewire.StageResponse{Metadata: metadata, Payload: payload}
	})
	defer stage.Close()

	now := time.Date(2026, 7, 21, 4, 0, 0, 0, time.UTC)
	member := deploymentSwitchMember(stage.URL, now)
	client := newBlockingDeploymentClient()
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{StageCount: 1}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)

	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a","stage_count":1,"ctx_size":256,"n_gpu_layers":0}`, http.StatusOK, "model-a", 1)

	firstInferenceDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		firstInferenceDone <- performLayerRun(server, "model-a", "active-request")
	}()
	select {
	case <-inferenceStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("active inference did not reach the stage")
	}

	client.blockLoad("model-b")
	switchDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		switchDone <- performSwitch(server, `{"deployment_id":"deployment-b","model":"model-b","stage_count":1,"ctx_size":256,"n_gpu_layers":0}`)
	}()
	waitForCoordinatorPhase(t, server, deploymentPhaseTransitioning)

	rejected := performLayerRun(server, "model-a", "rejected-during-drain")
	assertErrorCode(t, rejected, http.StatusServiceUnavailable, string(errorDeploymentTransitioning))

	close(releaseInference)
	if response := <-firstInferenceDone; response.Code != http.StatusOK {
		t.Fatalf("drained inference status=%d body=%s", response.Code, response.Body.String())
	}
	client.waitForLoad(t, "model-b")

	statusDuringLoad := readDeploymentStatus(t, server)
	if statusDuringLoad.Phase != deploymentPhaseTransitioning || statusDuringLoad.Active == nil || statusDuringLoad.Active.Model.ModelID != "model-a" {
		t.Fatalf("replacement was partially published during ready barrier: %+v", statusDuringLoad)
	}
	rejected = performLayerRun(server, "model-b", "rejected-during-load")
	assertErrorCode(t, rejected, http.StatusServiceUnavailable, string(errorDeploymentTransitioning))

	client.releaseLoad("model-b")
	response := <-switchDone
	if response.Code != http.StatusOK {
		t.Fatalf("model-b switch status=%d body=%s", response.Code, response.Body.String())
	}
	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || status.Active == nil || status.Active.Model.ModelID != "model-b" || status.Active.Epoch != 2 {
		t.Fatalf("unexpected active deployment after switch: %+v", status)
	}

	rejected = performLayerRun(server, "model-a", "old-model")
	assertErrorCode(t, rejected, http.StatusConflict, string(errorModelNotActive))
	if response := performLayerRun(server, "model-b", "new-model"); response.Code != http.StatusOK {
		t.Fatalf("new model inference status=%d body=%s", response.Code, response.Body.String())
	}

	client.failLoad("model-c")
	failed := performSwitch(server, `{"deployment_id":"deployment-c","model":"model-c","stage_count":1,"ctx_size":256,"n_gpu_layers":0}`)
	assertErrorCode(t, failed, http.StatusBadGateway, string(errorDeploymentSwitchFailed))
	status = readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseFailed || status.Active != nil || status.LastError == "" {
		t.Fatalf("failed transition left a publishable deployment: %+v", status)
	}
	rejected = performLayerRun(server, "model-b", "after-failed-transition")
	assertErrorCode(t, rejected, http.StatusServiceUnavailable, string(errorDeploymentUnavailable))
}

func deploymentSwitchRegistry() modelregistry.Registry {
	return modelregistry.Registry{Models: []cluster.ModelProfile{
		deploymentSwitchModel("model-a", strings.Repeat("a", 64)),
		deploymentSwitchModel("model-b", strings.Repeat("b", 64)),
		deploymentSwitchModel("model-c", strings.Repeat("c", 64)),
	}}
}

func deploymentSwitchModel(id, sha string) cluster.ModelProfile {
	return cluster.ModelProfile{
		ID:               id,
		Family:           "llm",
		SupportedEngines: []cluster.Engine{cluster.EngineLlamaCPP},
		LayerCount:       4,
		PlacementModes:   []cluster.ExecutionMode{cluster.ExecutionModePipelineParallel},
		ArtifactPath:     "/models/" + id + ".gguf",
		ArtifactSHA256:   sha,
	}
}

func deploymentSwitchMember(apiURL string, now time.Time) membership.Member {
	return membership.Member{
		ClusterID: "deployment-test",
		NodeID:    "node-a",
		NodeName:  "node-a",
		Hostname:  "host-a",
		APIURL:    apiURL,
		Arch:      "amd64",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:                64.0,
			cluster.CapabilityComputeBackends:         []string{"cpu"},
			cluster.CapabilityRuntimeEngine:           string(cluster.EngineLlamaCPP),
			cluster.CapabilityRuntimeComputeBackend:   string(cluster.ComputeBackendCPU),
			cluster.CapabilityRuntimeExecutionMode:    string(cluster.ExecutionModePipelineParallel),
			cluster.CapabilityRuntimeRevision:         "runtime-test",
			cluster.CapabilityRuntimeLlamaCPPRevision: "llama-test",
			cluster.CapabilityRuntimeCUDAActive:       false,
			cluster.CapabilityRuntimeStartsIdle:       true,
		},
		StartedAt: now.Add(-time.Hour),
		LastSeen:  now,
	}
}

func performSwitch(server *Server, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, api.PathDeploymentSwitch, strings.NewReader(body))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	return response
}

func performLayerRun(server *Server, model, requestID string) *httptest.ResponseRecorder {
	body := `{"model":"` + model + `","payload":"hello","max_tokens":1,"request_id":"` + requestID + `"}`
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitRun, strings.NewReader(body))
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	return response
}

func assertSwitchStatus(t *testing.T, server *Server, body string, wantStatus int, wantModel string, wantEpoch uint64) {
	t.Helper()
	response := performSwitch(server, body)
	if response.Code != wantStatus {
		t.Fatalf("switch status=%d body=%s", response.Code, response.Body.String())
	}
	status := readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Model.ModelID != wantModel || status.Active.Epoch != wantEpoch {
		t.Fatalf("unexpected active deployment: %+v", status)
	}
}

func assertErrorCode(t *testing.T, response *httptest.ResponseRecorder, wantStatus int, wantCode string) {
	t.Helper()
	if response.Code != wantStatus {
		t.Fatalf("status=%d want=%d body=%s", response.Code, wantStatus, response.Body.String())
	}
	var envelope struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Error != wantCode {
		t.Fatalf("error=%q want=%q body=%s", envelope.Error, wantCode, response.Body.String())
	}
}

func waitForCoordinatorPhase(t *testing.T, server *Server, want deploymentPhase) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if status := readDeploymentStatus(t, server); status.Phase == want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("coordinator did not reach phase %q", want)
}

func readDeploymentStatus(t *testing.T, server *Server) deploymentStatusResponse {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, api.PathDeploymentStatus, nil)
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("deployment status=%d body=%s", response.Code, response.Body.String())
	}
	var status deploymentStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	return status
}

type blockingDeploymentClient struct {
	mu          sync.Mutex
	status      runtimebridge.DeploymentStatus
	blockModel  string
	loadStarted chan string
	release     map[string]chan struct{}
	failModel   string
}

func newBlockingDeploymentClient() *blockingDeploymentClient {
	return &blockingDeploymentClient{
		status:      runtimebridge.DeploymentStatus{State: "idle"},
		loadStarted: make(chan string, 8),
		release:     make(map[string]chan struct{}),
	}
}

func (c *blockingDeploymentClient) Status(context.Context, string) (runtimebridge.DeploymentStatus, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return cloneRuntimeStatus(c.status), nil
}

func (c *blockingDeploymentClient) Load(ctx context.Context, _ string, request runtimebridge.LoadDeploymentRequest) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	fail := c.failModel == request.ModelID
	block := c.blockModel == request.ModelID
	release := c.release[request.ModelID]
	c.mu.Unlock()
	if fail {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected load failure")
	}
	if block {
		c.loadStarted <- request.ModelID
		select {
		case <-ctx.Done():
			return runtimebridge.DeploymentOperationResponse{}, ctx.Err()
		case <-release:
		}
	}
	status := runtimebridge.DeploymentStatus{
		Resident: true,
		Active:   false,
		State:    "ready",
		Deployment: &runtimebridge.DeploymentIdentity{
			DeploymentID: request.DeploymentID,
			ModelID:      request.ModelID,
		},
	}
	c.mu.Lock()
	c.status = status
	c.mu.Unlock()
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: cloneRuntimeStatus(status), Loaded: true}, nil
}

func (c *blockingDeploymentClient) Activate(_ context.Context, _ string, deploymentID string) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.status.Deployment == nil || c.status.Deployment.DeploymentID != deploymentID {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("deployment mismatch")
	}
	c.status.Active = true
	c.status.State = "active"
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: cloneRuntimeStatus(c.status), Activated: true}, nil
}

func (c *blockingDeploymentClient) Unload(_ context.Context, _ string, deploymentID string) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.status.Deployment == nil || c.status.Deployment.DeploymentID != deploymentID {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("deployment mismatch")
	}
	identity := *c.status.Deployment
	c.status = runtimebridge.DeploymentStatus{State: "idle"}
	return runtimebridge.DeploymentOperationResponse{
		DeploymentStatus: runtimebridge.DeploymentStatus{State: "idle", Deployment: &identity},
		Unloaded:         true,
	}, nil
}

func (c *blockingDeploymentClient) blockLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.blockModel = model
	c.release[model] = make(chan struct{})
}

func (c *blockingDeploymentClient) waitForLoad(t *testing.T, model string) {
	t.Helper()
	select {
	case got := <-c.loadStarted:
		if got != model {
			t.Fatalf("load started for %q, want %q", got, model)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("load for %q did not start", model)
	}
}

func (c *blockingDeploymentClient) releaseLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	close(c.release[model])
	c.blockModel = ""
}

func (c *blockingDeploymentClient) failLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failModel = model
}

func cloneRuntimeStatus(status runtimebridge.DeploymentStatus) runtimebridge.DeploymentStatus {
	if status.Deployment != nil {
		identity := *status.Deployment
		status.Deployment = &identity
	}
	return status
}
