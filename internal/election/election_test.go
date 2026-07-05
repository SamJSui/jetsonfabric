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

func assertLeader(t *testing.T, leader membership.Member, ok bool, nodeID string) {
	t.Helper()
	if !ok {
		t.Fatal("expected leader")
	}
	if leader.NodeID != nodeID {
		t.Fatalf("expected %s, got %s", nodeID, leader.NodeID)
	}
}

func testMember(id string, role membership.NodeRole, preference int, lastSeen time.Time, startedAt time.Time) membership.Member {
	return membership.Member{
		ClusterID:        "test-cluster",
		NodeID:           id,
		NodeName:         id,
		Role:             role,
		APIURL:           "http://" + id + ":52415",
		LeaderPreference: preference,
		ControlEligible:  membership.RoleLeaderEligible(role),
		StartedAt:        startedAt,
		LastSeen:         lastSeen,
	}
}

func testNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}
