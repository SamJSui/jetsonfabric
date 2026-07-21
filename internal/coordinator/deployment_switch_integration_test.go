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

func TestCoordinatorPublishesNewEpochWhileOldSessionsDrain(t *testing.T) {
	oldCalls := make(chan struct{}, 8)
	releaseOld := make(chan struct{})
	stage := newCoordinatorFrameServer(t, func(req stagewire.StageRequest) stagewire.StageResponse {
		if req.ModelID == "model-a" {
			oldCalls <- struct{}{}
			<-releaseOld
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
	client := newMultiDeploymentClient()
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{deploymentSwitchMember(stage.URL, now)}}, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{StageCount: 1}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)

	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a","stage_count":1,"ctx_size":256,"n_gpu_layers":0}`, http.StatusOK, "model-a", 1)
	oldDone := make(chan *httptest.ResponseRecorder, 2)
	go func() { oldDone <- performLayerRun(server, "model-a", "old-1") }()
	waitForSignal(t, oldCalls, "first old-epoch generation")
	go func() { oldDone <- performLayerRun(server, "model-a", "old-2") }()
	waitForSignal(t, oldCalls, "second old-epoch generation")

	client.blockLoad("model-b")
	switchDone := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		switchDone <- performSwitch(server, `{"deployment_id":"deployment-b","model":"model-b","stage_count":1,"ctx_size":256,"n_gpu_layers":0}`)
	}()
	client.waitForLoad(t, "model-b")
	waitForCoordinatorPhase(t, server, deploymentPhasePreparing)

	status := readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Epoch != 1 || status.InFlightByEpoch[1] != 2 {
		t.Fatalf("old epoch was not serving during prepare: %+v", status)
	}
	rejected := performLayerRun(server, "model-b", "too-early")
	assertErrorCode(t, rejected, http.StatusConflict, string(errorModelNotActive))

	client.releaseLoad("model-b")
	waitForCoordinatorPhase(t, server, deploymentPhaseDraining)
	status = readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Epoch != 2 || len(status.Draining) != 1 || status.Draining[0].Epoch != 1 {
		t.Fatalf("new epoch was not published beside the draining epoch: %+v", status)
	}
	if status.InFlightByEpoch[1] != 2 {
		t.Fatalf("old epoch admission count changed during publication: %+v", status.InFlightByEpoch)
	}
	if response := performLayerRun(server, "model-b", "new-during-drain"); response.Code != http.StatusOK {
		t.Fatalf("new epoch inference status=%d body=%s", response.Code, response.Body.String())
	}
	select {
	case response := <-switchDone:
		t.Fatalf("switch completed before old sessions drained: status=%d body=%s", response.Code, response.Body.String())
	case <-time.After(25 * time.Millisecond):
	}

	close(releaseOld)
	for range 2 {
		if response := <-oldDone; response.Code != http.StatusOK {
			t.Fatalf("old pinned generation status=%d body=%s", response.Code, response.Body.String())
		}
	}
	response := <-switchDone
	if response.Code != http.StatusOK {
		t.Fatalf("model-b switch status=%d body=%s", response.Code, response.Body.String())
	}
	status = readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || status.Active == nil || status.Active.Epoch != 2 || len(status.Draining) != 0 {
		t.Fatalf("old epoch did not finish cleanup: %+v", status)
	}
	if !client.operationBefore("load:model-b", "drain:model-a") {
		t.Fatal("old epoch drained before the replacement was prepared")
	}
	if !client.operationBefore("drain:model-a", "unload:model-a") {
		t.Fatal("old epoch unloaded before entering draining state")
	}
}

func TestFailedPrepareKeepsPreviousEpochServing(t *testing.T) {
	now := time.Date(2026, 7, 21, 4, 30, 0, 0, time.UTC)
	member := deploymentSwitchMember("http://node-a", now)
	client := newMultiDeploymentClient()
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{StageCount: 1}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)
	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a","stage_count":1}`, http.StatusOK, "model-a", 1)

	client.failLoad("model-b")
	failed := performSwitch(server, `{"deployment_id":"deployment-b","model":"model-b","stage_count":1}`)
	assertErrorCode(t, failed, http.StatusBadGateway, string(errorDeploymentSwitchFailed))
	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || status.Active == nil || status.Active.Epoch != 1 || status.LastError == "" {
		t.Fatalf("failed prepare displaced the previous epoch: %+v", status)
	}
	admission, err := server.deployments.admit("model-a")
	if err != nil {
		t.Fatalf("previous epoch was not admissible after rollback: %v", err)
	}
	admission.Release()
	if runtimeStatus := client.snapshot("http://node-a", runtimeDeploymentIdentity(*server.deployments.snapshot().Active)); !runtimeStatus.Active {
		t.Fatalf("previous runtime was not left active: %+v", runtimeStatus)
	}

	client.clearFailures()
	assertSwitchStatus(t, server, `{"deployment_id":"deployment-b-retry","model":"model-b","stage_count":1}`, http.StatusOK, "model-b", 3)
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
		ID: id, Family: "llm", SupportedEngines: []cluster.Engine{cluster.EngineLlamaCPP},
		LayerCount: 4, PlacementModes: []cluster.ExecutionMode{cluster.ExecutionModePipelineParallel},
		ArtifactPath: "/models/" + id + ".gguf", ArtifactSHA256: sha,
	}
}

func deploymentSwitchMember(apiURL string, now time.Time) membership.Member {
	return membership.Member{
		ClusterID: "deployment-test", NodeID: "node-a", NodeName: "node-a",
		Hostname: "host-a", APIURL: apiURL, Arch: "amd64",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 64.0, cluster.CapabilityComputeBackends: []string{"cpu"},
			cluster.CapabilityRuntimeEngine:           string(cluster.EngineLlamaCPP),
			cluster.CapabilityRuntimeComputeBackend:   string(cluster.ComputeBackendCPU),
			cluster.CapabilityRuntimeExecutionMode:    string(cluster.ExecutionModePipelineParallel),
			cluster.CapabilityRuntimeRevision:         "runtime-test",
			cluster.CapabilityRuntimeLlamaCPPRevision: "llama-test",
			cluster.CapabilityRuntimeCUDAActive:       false,
			cluster.CapabilityRuntimeStartsIdle:       true,
		},
		StartedAt: now.Add(-time.Hour), LastSeen: now,
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

func waitForSignal(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
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

type runtimeDeploymentKey struct {
	DeploymentID string
	Epoch        uint64
}

type fakeRuntimeDeployments struct {
	deployments map[runtimeDeploymentKey]runtimebridge.DeploymentStatus
	preferred   runtimeDeploymentKey
}

type multiDeploymentClient struct {
	mu              sync.Mutex
	runtimes        map[string]*fakeRuntimeDeployments
	blockModel      string
	loadStarted     chan string
	releaseLoadByID map[string]chan struct{}
	failLoadModel   string
	failActivateURL string
	failUnload      bool
	unreachable     map[string]bool
	operations      []string
}

func newMultiDeploymentClient() *multiDeploymentClient {
	return &multiDeploymentClient{
		runtimes: make(map[string]*fakeRuntimeDeployments), loadStarted: make(chan string, 8),
		releaseLoadByID: make(map[string]chan struct{}), unreachable: make(map[string]bool),
	}
}

func (c *multiDeploymentClient) Status(_ context.Context, nodeURL string) (runtimebridge.DeploymentStatus, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.unreachable[nodeURL] {
		return runtimebridge.DeploymentStatus{}, errors.New("injected node loss")
	}
	runtime := c.runtime(nodeURL)
	return cloneRuntimeStatus(runtime.deployments[runtime.preferred]), nil
}

func (c *multiDeploymentClient) Load(ctx context.Context, nodeURL string, request runtimebridge.LoadDeploymentRequest) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	fail := c.failLoadModel == request.ModelID
	unreachable := c.unreachable[nodeURL]
	block := c.blockModel == request.ModelID
	release := c.releaseLoadByID[request.ModelID]
	c.operations = append(c.operations, "load:"+request.ModelID)
	c.mu.Unlock()
	if unreachable {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected node loss")
	}
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
	status := readyRuntimeStatus(request)
	c.mu.Lock()
	runtime := c.runtime(nodeURL)
	key := keyForRuntimeIdentity(*status.Deployment)
	runtime.deployments[key] = status
	if len(runtime.deployments) == 1 {
		runtime.preferred = key
	}
	c.mu.Unlock()
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: cloneRuntimeStatus(status), Loaded: true}, nil
}

func (c *multiDeploymentClient) Activate(_ context.Context, nodeURL string, identity runtimebridge.DeploymentIdentity) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.operations = append(c.operations, "activate:"+identity.ModelID)
	if c.unreachable[nodeURL] {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected node loss")
	}
	if nodeURL == c.failActivateURL {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected activation failure")
	}
	runtime := c.runtime(nodeURL)
	key := keyForRuntimeIdentity(identity)
	status, ok := runtime.deployments[key]
	if !ok || !runtimeDeploymentIdentityMatches(status.Deployment, identity) {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("deployment not found")
	}
	status.Active, status.State, status.ModelMemory.Pinned = true, "active", true
	runtime.deployments[key], runtime.preferred = status, key
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: cloneRuntimeStatus(status), Activated: true}, nil
}

func (c *multiDeploymentClient) Drain(_ context.Context, nodeURL string, identity runtimebridge.DeploymentIdentity) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.operations = append(c.operations, "drain:"+identity.ModelID)
	if c.unreachable[nodeURL] {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected node loss")
	}
	runtime := c.runtime(nodeURL)
	key := keyForRuntimeIdentity(identity)
	status, ok := runtime.deployments[key]
	if !ok {
		return idleOperationResponse(identity, "drain"), nil
	}
	if status.State == "active" {
		status.State = "draining"
		runtime.deployments[key] = status
	}
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: cloneRuntimeStatus(status), Drained: true}, nil
}

func (c *multiDeploymentClient) Unload(_ context.Context, nodeURL string, identity runtimebridge.DeploymentIdentity) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.operations = append(c.operations, "unload:"+identity.ModelID)
	if c.unreachable[nodeURL] {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected node loss")
	}
	if c.failUnload {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected unload failure")
	}
	runtime := c.runtime(nodeURL)
	key := keyForRuntimeIdentity(identity)
	status, ok := runtime.deployments[key]
	if ok && status.State == "active" {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("active deployment was not drained")
	}
	delete(runtime.deployments, key)
	if runtime.preferred == key {
		runtime.preferred = newestRuntimeKey(runtime.deployments)
	}
	return idleOperationResponse(identity, "unload"), nil
}

func (c *multiDeploymentClient) runtime(nodeURL string) *fakeRuntimeDeployments {
	runtime := c.runtimes[nodeURL]
	if runtime == nil {
		runtime = &fakeRuntimeDeployments{deployments: make(map[runtimeDeploymentKey]runtimebridge.DeploymentStatus)}
		c.runtimes[nodeURL] = runtime
	}
	return runtime
}

func (c *multiDeploymentClient) snapshot(nodeURL string, identity runtimebridge.DeploymentIdentity) runtimebridge.DeploymentStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return cloneRuntimeStatus(c.runtime(nodeURL).deployments[keyForRuntimeIdentity(identity)])
}

func (c *multiDeploymentClient) blockLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.blockModel = model
	c.releaseLoadByID[model] = make(chan struct{})
}

func (c *multiDeploymentClient) waitForLoad(t *testing.T, model string) {
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

func (c *multiDeploymentClient) releaseLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	close(c.releaseLoadByID[model])
	c.blockModel = ""
}

func (c *multiDeploymentClient) failLoad(model string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failLoadModel = model
}

func (c *multiDeploymentClient) clearFailures() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failLoadModel, c.failActivateURL, c.failUnload = "", "", false
}

func (c *multiDeploymentClient) setUnreachable(nodeURL string, unreachable bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.unreachable[nodeURL] = unreachable
}

func (c *multiDeploymentClient) operationBefore(first, second string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	firstIndex, secondIndex := -1, -1
	for index, operation := range c.operations {
		if operation == first && firstIndex < 0 {
			firstIndex = index
		}
		if operation == second && secondIndex < 0 {
			secondIndex = index
		}
	}
	return firstIndex >= 0 && secondIndex > firstIndex
}

func readyRuntimeStatus(request runtimebridge.LoadDeploymentRequest) runtimebridge.DeploymentStatus {
	return runtimebridge.DeploymentStatus{
		Resident: true, State: "ready",
		Deployment: &runtimebridge.DeploymentIdentity{
			DeploymentID: request.DeploymentID, Epoch: request.Epoch,
			ModelID: request.ModelID, ModelSHA256: request.ModelSHA256,
		},
		ModelMemory: deploymentTestModelMemory(request, false),
	}
}

func idleOperationResponse(identity runtimebridge.DeploymentIdentity, operation string) runtimebridge.DeploymentOperationResponse {
	response := runtimebridge.DeploymentOperationResponse{
		DeploymentStatus: runtimebridge.DeploymentStatus{State: "idle", Deployment: &identity},
	}
	if operation == "drain" {
		response.Drained = true
	} else {
		response.Unloaded = true
	}
	return response
}

func keyForRuntimeIdentity(identity runtimebridge.DeploymentIdentity) runtimeDeploymentKey {
	return runtimeDeploymentKey{DeploymentID: identity.DeploymentID, Epoch: identity.Epoch}
}

func newestRuntimeKey(deployments map[runtimeDeploymentKey]runtimebridge.DeploymentStatus) runtimeDeploymentKey {
	var newest runtimeDeploymentKey
	for key := range deployments {
		if key.Epoch > newest.Epoch {
			newest = key
		}
	}
	return newest
}

func cloneRuntimeStatus(status runtimebridge.DeploymentStatus) runtimebridge.DeploymentStatus {
	if status.Deployment != nil {
		identity := *status.Deployment
		status.Deployment = &identity
	}
	if status.ModelMemory != nil {
		memory := *status.ModelMemory
		status.ModelMemory = &memory
	}
	return status
}

func deploymentTestModelMemory(request runtimebridge.LoadDeploymentRequest, pinned bool) *runtimebridge.ModelMemory {
	const layerCount, totalWeightBytes = 4, 400
	assignedLayers := request.LayerEnd - request.LayerStart
	return &runtimebridge.ModelMemory{
		LayerStart: request.LayerStart, LayerEnd: request.LayerEnd, LayerCount: layerCount,
		ResidentWeightBytes: uint64(assignedLayers * (totalWeightBytes / layerCount)),
		TotalWeightBytes:    totalWeightBytes, ResidentTensorCount: uint64(assignedLayers),
		Partitioned: request.LayerStart != 0 || request.LayerEnd != layerCount, Pinned: pinned,
	}
}
