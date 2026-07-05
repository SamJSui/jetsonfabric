package election

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestElectLeaderPrefersCoordinatorRole(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("jetson", membership.NodeRoleJetson, 100, now, now.Add(-time.Hour)),
		testMember("coord", membership.NodeRoleCoordinator, 0, now, now),
	}, time.Minute)

	assertLeader(t, leader, ok, "coord")
}

func TestElectLeaderSkipsWorkerAndTestRoles(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("worker", membership.NodeRoleWorker, 100, now, now.Add(-time.Hour)),
		testMember("test", membership.NodeRoleTest, 100, now, now.Add(-time.Hour)),
		testMember("jetson", membership.NodeRoleJetson, 0, now, now),
	}, time.Minute)

	assertLeader(t, leader, ok, "jetson")
}

func TestElectLeaderUsesPreferenceWithinSameRole(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("low", membership.NodeRoleJetson, 1, now, now.Add(-time.Hour)),
		testMember("high", membership.NodeRoleJetson, 5, now, now),
	}, time.Minute)

	assertLeader(t, leader, ok, "high")
}

func TestElectLeaderPrefersOlderPeerWithinSameRank(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("new", membership.NodeRoleJetson, 0, now, now),
		testMember("old", membership.NodeRoleJetson, 0, now, now.Add(-time.Hour)),
	}, time.Minute)

	assertLeader(t, leader, ok, "old")
}

func TestElectLeaderUsesStableTieBreak(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("node-z", membership.NodeRoleJetson, 0, now, now),
		testMember("node-a", membership.NodeRoleJetson, 0, now, now),
	}, time.Minute)

	assertLeader(t, leader, ok, "node-a")
}

func TestElectLeaderSkipsStaleMembers(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("stale", membership.NodeRoleCoordinator, 0, now.Add(-2*time.Minute), now),
		testMember("fresh", membership.NodeRoleJetson, 0, now, now),
	}, time.Minute)

	assertLeader(t, leader, ok, "fresh")
}

func TestExplainReportsCandidateReasons(t *testing.T) {
	now := testNow()
	result := Explain(now, []membership.Member{
		testMember("worker", membership.NodeRoleWorker, 0, now, now),
		testMember("stale", membership.NodeRoleCoordinator, 0, now.Add(-2*time.Minute), now),
		testMember("leader", membership.NodeRoleJetson, 0, now, now),
	}, time.Minute)

	if result.Leader == nil || result.Leader.NodeID != "leader" {
		t.Fatalf("unexpected leader: %+v", result.Leader)
	}
	assertReason(t, result, "worker", ReasonRoleNotEligible)
	assertReason(t, result, "stale", ReasonStaleMember)
}

func TestTrackerKeepsIncumbentDuringLease(t *testing.T) {
	now := testNow()
	tracker := NewTracker(10 * time.Second)
	members := []membership.Member{
		testMember("incumbent", membership.NodeRoleJetson, 0, now, now),
	}
	result := tracker.Explain(now, members, time.Minute)
	assertResultLeader(t, result, "incumbent")

	members = append(members, testMember("older", membership.NodeRoleJetson, 0, now, now.Add(-time.Hour)))
	result = tracker.Explain(now.Add(time.Second), members, time.Minute)
	assertResultLeader(t, result, "incumbent")
	if result.Epoch != 1 {
		t.Fatalf("expected epoch 1, got %d", result.Epoch)
	}
}

func TestTrackerAllowsHigherRankPreemption(t *testing.T) {
	now := testNow()
	tracker := NewTracker(10 * time.Second)
	first := []membership.Member{testMember("jetson", membership.NodeRoleJetson, 0, now, now)}
	assertResultLeader(t, tracker.Explain(now, first, time.Minute), "jetson")

	second := append(first, testMember("coord", membership.NodeRoleCoordinator, 0, now, now))
	result := tracker.Explain(now.Add(time.Second), second, time.Minute)
	assertResultLeader(t, result, "coord")
	if result.Epoch != 2 {
		t.Fatalf("expected epoch 2 after preemption, got %d", result.Epoch)
	}
}

func TestTrackerFailsOverWhenIncumbentIsStale(t *testing.T) {
	now := testNow()
	tracker := NewTracker(10 * time.Second)
	first := []membership.Member{testMember("old", membership.NodeRoleJetson, 0, now, now)}
	assertResultLeader(t, tracker.Explain(now, first, time.Minute), "old")

	second := []membership.Member{
		testMember("old", membership.NodeRoleJetson, 0, now.Add(-2*time.Minute), now),
		testMember("fresh", membership.NodeRoleJetson, 0, now, now),
	}
	assertResultLeader(t, tracker.Explain(now.Add(time.Second), second, time.Minute), "fresh")
}

func assertLeader(t *testing.T, leader membership.Member, ok bool, nodeID string) {
	t.Helper()
	if !ok {
		t.Fatal("expected leader")
	}
	if leader.NodeID != nodeID {
		t.Fatalf("expected %s, got %s", nodeID, leader.NodeID)
	}
}

func assertResultLeader(t *testing.T, result Result, nodeID string) {
	t.Helper()
	if result.Leader == nil {
		t.Fatal("expected leader")
	}
	if result.Leader.NodeID != nodeID {
		t.Fatalf("expected %s, got %s", nodeID, result.Leader.NodeID)
	}
}

func assertReason(t *testing.T, result Result, nodeID string, reason string) {
	t.Helper()
	for _, candidate := range result.Candidates {
		if candidate.Member.NodeID == nodeID && candidate.Reason == reason {
			return
		}
	}
	t.Fatalf("expected %s reason %s in %+v", nodeID, reason, result.Candidates)
}

func testMember(id string, role membership.NodeRole, preference int, lastSeen time.Time, startedAt time.Time) membership.Member {
	return membership.Member{
		ClusterID:        "test-cluster",
		NodeID:           id,
		NodeName:         id,
		Role:             role,
		APIURL:           "http://" + id + ":52415",
		LeaderPreference: preference,
		StartedAt:        startedAt,
		LastSeen:         lastSeen,
	}
}

func testNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}
