package coordinator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func TestCoordinatorRollsBackPartialActivationWithoutPublishing(t *testing.T) {
	nodeA := httptest.NewServer(http.NotFoundHandler())
	defer nodeA.Close()
	nodeB := httptest.NewServer(http.NotFoundHandler())
	defer nodeB.Close()

	now := time.Date(2026, 7, 21, 5, 0, 0, 0, time.UTC)
	members := []membership.Member{
		partialFailureMember("node-a", "host-a", nodeA.URL, now),
		partialFailureMember("node-b", "host-b", nodeB.URL, now),
	}
	client := newMultiDeploymentClient()
	client.failActivateURL = nodeB.URL
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
	if status.Phase != deploymentPhaseFailed || status.Active != nil || status.Preparing != nil || status.LastError == "" {
		t.Fatalf("partial activation published an unusable epoch: %+v", status)
	}

	identity := runtimeIdentity("deployment-partial", 1, "model-a", 'a')
	for _, nodeURL := range []string{nodeA.URL, nodeB.URL} {
		if runtimeStatus := client.snapshot(nodeURL, identity); runtimeStatus.Resident || runtimeStatus.Active {
			t.Fatalf("runtime %s retained the failed epoch: %+v", nodeURL, runtimeStatus)
		}
	}
	if !client.operationBefore("activate:model-a", "drain:model-a") {
		t.Fatal("rollback did not drain the partially activated epoch")
	}
}

func TestCoordinatorRetriesOldEpochCleanupAfterPublication(t *testing.T) {
	now := time.Date(2026, 7, 21, 5, 30, 0, 0, time.UTC)
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

	client.failUnload = true
	response := performSwitch(server, `{"deployment_id":"deployment-b","model":"model-b","stage_count":1}`)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "cleanup_error") {
		t.Fatalf("published switch did not report deferred cleanup: status=%d body=%s", response.Code, response.Body.String())
	}
	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseDraining || status.Active == nil || status.Active.Epoch != 2 || len(status.Draining) != 1 {
		t.Fatalf("cleanup failure displaced the published epoch: %+v", status)
	}
	if err := server.Reconcile(context.Background()); err == nil {
		t.Fatal("persistent cleanup failure was reported as reconciled")
	}
	status = readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseDraining || status.LastError == "" {
		t.Fatalf("pending cleanup error disappeared from status: %+v", status)
	}

	client.clearFailures()
	if err := server.Reconcile(context.Background()); err != nil {
		t.Fatalf("cleanup retry failed: %v", err)
	}
	status = readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || status.Active == nil || status.Active.Epoch != 2 || len(status.Draining) != 0 {
		t.Fatalf("cleanup retry did not converge: %+v", status)
	}
}

func TestCoordinatorRollbackPreservesEarlierDrainingEpoch(t *testing.T) {
	now := time.Date(2026, 7, 21, 5, 40, 0, 0, time.UTC)
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

	client.failUnload = true
	response := performSwitch(server, `{"deployment_id":"deployment-b","model":"model-b","stage_count":1}`)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "cleanup_error") {
		t.Fatalf("replacement did not retain failed cleanup: status=%d body=%s", response.Code, response.Body.String())
	}

	client.clearFailures()
	client.failLoad("model-c")
	failed := performSwitch(server, `{"deployment_id":"deployment-c","model":"model-c","stage_count":1}`)
	assertErrorCode(t, failed, http.StatusBadGateway, string(errorDeploymentSwitchFailed))
	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseDraining || status.Active == nil || status.Active.Epoch != 2 || len(status.Draining) != 1 || status.Draining[0].Epoch != 1 {
		t.Fatalf("rollback hid the earlier draining epoch: %+v", status)
	}
}

func TestCoordinatorPrepareTimeoutRollsBackToActiveEpoch(t *testing.T) {
	now := time.Date(2026, 7, 21, 5, 45, 0, 0, time.UTC)
	member := deploymentSwitchMember("http://node-a", now)
	client := newMultiDeploymentClient()
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{member}}, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{StageCount: 1}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
		WithDeploymentTimeouts(30*time.Millisecond, time.Second),
	)
	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a","stage_count":1}`, http.StatusOK, "model-a", 1)

	client.blockLoad("model-b")
	response := performSwitch(server, `{"deployment_id":"deployment-timeout","model":"model-b","stage_count":1}`)
	assertErrorCode(t, response, http.StatusRequestTimeout, string(errorDeploymentSwitchFailed))
	status := readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || status.Active == nil || status.Active.Epoch != 1 || status.LastError == "" {
		t.Fatalf("prepare timeout displaced the active epoch: %+v", status)
	}
}

func partialFailureMember(nodeID, hostname, apiURL string, now time.Time) membership.Member {
	member := deploymentSwitchMember(apiURL, now)
	member.NodeID = nodeID
	member.NodeName = nodeID
	member.Hostname = hostname
	return member
}

func runtimeIdentity(deploymentID string, epoch uint64, modelID string, digest byte) runtimebridge.DeploymentIdentity {
	return runtimebridge.DeploymentIdentity{
		DeploymentID: deploymentID,
		Epoch:        epoch,
		ModelID:      modelID,
		ModelSHA256:  strings.Repeat(string(digest), 64),
	}
}
