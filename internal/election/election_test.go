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

func TestElectLeaderReturnsFalseWhenNoEligibleMembers(t *testing.T) {
	now := testNow()
	leader, ok := ElectLeader(now, []membership.Member{
		testMember("worker", membership.NodeRoleWorker, 0, now, now),
		testMember("test", membership.NodeRoleTest, 0, now, now),
		testMember("stale-jetson", membership.NodeRoleJetson, 0, now.Add(-2*time.Minute), now),
	}, time.Minute)

	if ok {
		t.Fatalf("expected no leader, got %+v", leader)
	}
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

func TestSameMembershipViewElectsSameLeaderRegardlessOfOrder(t *testing.T) {
	now := testNow()

	a := testMember("node-a", membership.NodeRoleJetson, 0, now, now)
	b := testMember("node-b", membership.NodeRoleJetson, 0, now, now.Add(-time.Minute))
	c := testMember("node-c", membership.NodeRoleWorker, 100, now, now.Add(-time.Hour))

	result1 := Explain(now, []membership.Member{a, b, c}, time.Minute)
	result2 := Explain(now, []membership.Member{c, a, b}, time.Minute)
	result3 := Explain(now, []membership.Member{b, c, a}, time.Minute)

	if result1.Leader == nil || result2.Leader == nil || result3.Leader == nil {
		t.Fatalf("expected leaders: %+v %+v %+v", result1.Leader, result2.Leader, result3.Leader)
	}

	want := result1.Leader.NodeID
	if result2.Leader.NodeID != want || result3.Leader.NodeID != want {
		t.Fatalf("leaders diverged: %s, %s, %s", result1.Leader.NodeID, result2.Leader.NodeID, result3.Leader.NodeID)
	}
}

func TestTrackerRecomputesLeaderDuringLease(t *testing.T) {
	now := testNow()
	tracker := NewTracker(10 * time.Second)

	members := []membership.Member{
		testMember("incumbent", membership.NodeRoleJetson, 0, now, now),
	}

	result := tracker.Explain(now, members, time.Minute)
	assertResultLeader(t, result, "incumbent")

	members = append(members, testMember("older", membership.NodeRoleJetson, 0, now, now.Add(-time.Hour)))

	result = tracker.Explain(now.Add(time.Second), members, time.Minute)
	assertResultLeader(t, result, "older")

	if result.Epoch != 2 {
		t.Fatalf("expected epoch 2 after deterministic leader change, got %d", result.Epoch)
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

func TestTrackerEpochDoesNotChangeWhenLeaderStaysSame(t *testing.T) {
	now := testNow()
	tracker := NewTracker(10 * time.Second)
	members := []membership.Member{testMember("node-a", membership.NodeRoleJetson, 0, now, now)}

	first := tracker.Explain(now, members, time.Minute)
	assertResultLeader(t, first, "node-a")
	if first.Epoch != 1 {
		t.Fatalf("expected first epoch 1, got %d", first.Epoch)
	}

	second := tracker.Explain(now.Add(time.Second), members, time.Minute)
	assertResultLeader(t, second, "node-a")
	if second.Epoch != 1 {
		t.Fatalf("expected epoch to remain 1, got %d", second.Epoch)
	}
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
