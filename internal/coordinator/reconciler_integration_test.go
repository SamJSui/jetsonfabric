package coordinator

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestReconcilerExpandsAndReplacesCapacityAutomatically(t *testing.T) {
	now := time.Date(2026, 7, 21, 6, 0, 0, 0, time.UTC)
	nodeA := deploymentSwitchMember("http://node-a", now)
	nodeB := partialFailureMember("node-b", "host-b", "http://node-b", now)
	source := newMutableMemberSource(nodeA)
	client := newMultiDeploymentClient()
	registry := deploymentSwitchRegistry()
	for index := range registry.Models {
		registry.Models[index].MinMemoryGB = 8
	}
	server := NewServer(
		registry,
		WithMembershipSource(source, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)

	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a"}`, 200, "model-a", 1)
	if got := len(server.deployments.snapshot().Active.Stages()); got != 1 {
		t.Fatalf("initial auto placement stages=%d, want 1", got)
	}

	source.Set(nodeA, nodeB)
	if err := server.Reconcile(context.Background()); err != nil {
		t.Fatalf("add-node reconciliation failed: %v", err)
	}
	status := readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Epoch != 2 || len(status.Active.Stages) != 2 {
		t.Fatalf("add-node reconciliation did not expand the deployment: %+v", status)
	}

	nodeA.Capabilities[cluster.CapabilityMemoryGB] = 4.0
	source.Set(nodeA, nodeB)
	if err := server.Reconcile(context.Background()); err != nil {
		t.Fatalf("capacity reconciliation failed: %v", err)
	}
	status = readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Epoch != 3 || len(status.Active.Stages) != 1 || status.Active.Stages[0].NodeID != "node-b" {
		t.Fatalf("capacity change did not replace the ineligible node: %+v", status)
	}
}

func TestReconcilerPublishesAroundNodeLossAndCleansOnReturn(t *testing.T) {
	now := time.Date(2026, 7, 21, 6, 30, 0, 0, time.UTC)
	nodeA := deploymentSwitchMember("http://node-a", now)
	nodeB := partialFailureMember("node-b", "host-b", "http://node-b", now)
	source := newMutableMemberSource(nodeA, nodeB)
	client := newMultiDeploymentClient()
	server := NewServer(
		deploymentSwitchRegistry(),
		WithMembershipSource(source, time.Minute),
		WithClusterPlanPolicy(clusterplan.Policy{}),
		WithClock(func() time.Time { return now }),
		WithDeploymentClient(client),
	)
	assertSwitchStatus(t, server, `{"deployment_id":"deployment-a","model":"model-a"}`, 200, "model-a", 1)

	source.Set(nodeB)
	client.setUnreachable(nodeA.APIURL, true)
	if err := server.Reconcile(context.Background()); err == nil {
		t.Fatal("node-loss reconciliation omitted the deferred cleanup error")
	}
	status := readDeploymentStatus(t, server)
	if status.Active == nil || status.Active.Epoch != 2 || len(status.Active.Stages) != 1 || status.Active.Stages[0].NodeID != "node-b" {
		t.Fatalf("replacement epoch was not published around the lost node: %+v", status)
	}
	if status.Phase != deploymentPhaseDraining || len(status.Draining) != 1 {
		t.Fatalf("lost-node cleanup was not retained for retry: %+v", status)
	}
	admission, err := server.deployments.admit("model-a")
	if err != nil || admission.Plan == nil || admission.Plan.Identity().Epoch != 2 {
		t.Fatalf("healthy replacement was not admissible during deferred cleanup: plan=%v err=%v", admission.Plan, err)
	}
	admission.Release()

	client.setUnreachable(nodeA.APIURL, false)
	if err := server.Reconcile(context.Background()); err != nil {
		t.Fatalf("returned-node cleanup failed: %v", err)
	}
	status = readDeploymentStatus(t, server)
	if status.Phase != deploymentPhaseActive || len(status.Draining) != 0 || status.Active == nil || status.Active.Epoch != 2 {
		t.Fatalf("returned-node cleanup did not converge: %+v", status)
	}
}

type mutableMemberSource struct {
	mu      sync.RWMutex
	members []membership.Member
}

func newMutableMemberSource(members ...membership.Member) *mutableMemberSource {
	source := &mutableMemberSource{}
	source.Set(members...)
	return source
}

func (s *mutableMemberSource) Set(members ...membership.Member) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.members = append([]membership.Member(nil), members...)
}

func (s *mutableMemberSource) List() []membership.Member {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]membership.Member(nil), s.members...)
}
