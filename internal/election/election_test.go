package election

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestElectLeaderChoosesHighestPriority(t *testing.T) {
	now := time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
	leader, ok := ElectLeader(now, []membership.Member{
		member("a", 10, now),
		member("b", 50, now),
		member("c", 20, now),
	}, time.Minute)
	if !ok {
		t.Fatal("expected leader")
	}
	if leader.NodeID != "b" {
		t.Fatalf("expected b, got %s", leader.NodeID)
	}
}

func TestElectLeaderUsesStableTieBreak(t *testing.T) {
	now := time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
	leader, ok := ElectLeader(now, []membership.Member{
		member("node-z", 10, now),
		member("node-a", 10, now),
	}, time.Minute)
	if !ok {
		t.Fatal("expected leader")
	}
	if leader.NodeID != "node-a" {
		t.Fatalf("expected node-a, got %s", leader.NodeID)
	}
}

func TestElectLeaderSkipsStaleMembers(t *testing.T) {
	now := time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
	leader, ok := ElectLeader(now, []membership.Member{
		member("stale", 100, now.Add(-2*time.Minute)),
		member("fresh", 10, now),
	}, time.Minute)
	if !ok {
		t.Fatal("expected leader")
	}
	if leader.NodeID != "fresh" {
		t.Fatalf("expected fresh, got %s", leader.NodeID)
	}
}

func member(nodeID string, priority int, lastSeen time.Time) membership.Member {
	return membership.Member{
		ClusterID:       "test-cluster",
		NodeID:          nodeID,
		NodeName:        nodeID,
		APIURL:          "http://" + nodeID + ":52415",
		ControlEligible: true,
		ControlPriority: priority,
		LastSeen:        lastSeen,
	}
}
