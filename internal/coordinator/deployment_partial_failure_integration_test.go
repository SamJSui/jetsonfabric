package coordinator

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func TestCoordinatorDeploymentSwitchCleansPartialActivationFailure(t *testing.T) {
	nodeA := httptest.NewServer(http.NotFoundHandler())
	defer nodeA.Close()
	nodeB := httptest.NewServer(http.NotFoundHandler())
	defer nodeB.Close()

	now := time.Date(2026, 7, 21, 5, 0, 0, 0, time.UTC)
	members := []membership.Member{
		partialFailureMember("node-a", "host-a", nodeA.URL, now),
		partialFailureMember("node-b", "host-b", nodeB.URL, now),
	}
	client := newPartialFailureDeploymentClient(nodeB.URL)
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(staticMemberSource{members: members}, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{StageCount: 2}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)

	response := performSwitch(server, `{"deployment_id":"deployment-partial","model":"model-a","stage_count":2,"ctx_size":256,"n_gpu_layers":0}`)
	assertErrorCode(t, response, http.StatusBadGateway, string(errorDeploymentSwitchFailed))

	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseFailed || status.Active != nil || status.LastError == "" {
		t.Fatalf("partial activation failure published a deployment: %+v", status)
	}

	for _, nodeURL := range []string{nodeA.URL, nodeB.URL} {
		runtimeStatus, unloadCount := client.snapshot(nodeURL)
		if runtimeStatus.Resident || runtimeStatus.Active || runtimeStatus.State != "idle" {
			t.Fatalf("runtime %s was not cleaned after partial activation: %+v", nodeURL, runtimeStatus)
		}
		if unloadCount != 1 {
			t.Fatalf("runtime %s unload count=%d, want 1", nodeURL, unloadCount)
		}
	}
	if !client.wasActivated(nodeA.URL) {
		t.Fatal("first runtime never activated, so the test did not exercise partial activation")
	}
	if client.wasActivated(nodeB.URL) {
		t.Fatal("failing runtime was incorrectly recorded as activated")
	}
}

func partialFailureMember(nodeID, hostname, apiURL string, now time.Time) membership.Member {
	return membership.Member{
		ClusterID: "deployment-partial-failure",
		NodeID:    nodeID,
		NodeName:  nodeID,
		Hostname:  hostname,
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

type partialFailureDeploymentClient struct {
	mu              sync.Mutex
	statuses        map[string]runtimebridge.DeploymentStatus
	failActivateURL string
	unloadCounts    map[string]int
	activated       map[string]bool
}

func newPartialFailureDeploymentClient(failActivateURL string) *partialFailureDeploymentClient {
	return &partialFailureDeploymentClient{
		statuses:        make(map[string]runtimebridge.DeploymentStatus),
		failActivateURL: failActivateURL,
		unloadCounts:    make(map[string]int),
		activated:       make(map[string]bool),
	}
}

func (c *partialFailureDeploymentClient) Status(_ context.Context, nodeURL string) (runtimebridge.DeploymentStatus, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return copyDeploymentStatus(c.statuses[nodeURL]), nil
}

func (c *partialFailureDeploymentClient) Load(_ context.Context, nodeURL string, request runtimebridge.LoadDeploymentRequest) (runtimebridge.DeploymentOperationResponse, error) {
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
	c.statuses[nodeURL] = status
	c.mu.Unlock()
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: copyDeploymentStatus(status), Loaded: true}, nil
}

func (c *partialFailureDeploymentClient) Activate(_ context.Context, nodeURL string, deploymentID string) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	status := c.statuses[nodeURL]
	if status.Deployment == nil || status.Deployment.DeploymentID != deploymentID {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("deployment mismatch")
	}
	if nodeURL == c.failActivateURL {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("injected activation failure")
	}
	status.Active = true
	status.State = "active"
	c.statuses[nodeURL] = status
	c.activated[nodeURL] = true
	return runtimebridge.DeploymentOperationResponse{DeploymentStatus: copyDeploymentStatus(status), Activated: true}, nil
}

func (c *partialFailureDeploymentClient) Unload(_ context.Context, nodeURL string, deploymentID string) (runtimebridge.DeploymentOperationResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	status := c.statuses[nodeURL]
	if status.Deployment == nil || status.Deployment.DeploymentID != deploymentID {
		return runtimebridge.DeploymentOperationResponse{}, errors.New("deployment mismatch")
	}
	identity := *status.Deployment
	c.statuses[nodeURL] = runtimebridge.DeploymentStatus{State: "idle"}
	c.unloadCounts[nodeURL]++
	return runtimebridge.DeploymentOperationResponse{
		DeploymentStatus: runtimebridge.DeploymentStatus{State: "idle", Deployment: &identity},
		Unloaded:         true,
	}, nil
}

func (c *partialFailureDeploymentClient) snapshot(nodeURL string) (runtimebridge.DeploymentStatus, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return copyDeploymentStatus(c.statuses[nodeURL]), c.unloadCounts[nodeURL]
}

func (c *partialFailureDeploymentClient) wasActivated(nodeURL string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.activated[nodeURL]
}

func copyDeploymentStatus(status runtimebridge.DeploymentStatus) runtimebridge.DeploymentStatus {
	copy := status
	if status.Deployment != nil {
		identity := *status.Deployment
		copy.Deployment = &identity
	}
	return copy
}
