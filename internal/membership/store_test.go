package membership

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestStoreUpsertIgnoresInvalidMember(t *testing.T) {
	store := NewStore()

	store.Upsert(Member{
		ClusterID: "home-lab",
		NodeID:    "",
		APIURL:    "http://dopey.local:52415",
	})

	if members := store.List(); len(members) != 0 {
		t.Fatalf("expected invalid member to be ignored, got %+v", members)
	}
}

func TestStorePreservesRichFieldsOnLightweightUpdate(t *testing.T) {
	store := NewStore()
	initial := memberWithRuntimeDetails()
	store.Upsert(initial)

	lightweight := initial
	lightweight.Capabilities = nil
	lightweight.Metrics = nil
	lightweight.Engines = nil
	lightweight.APIURL = "http://node.local:52415"
	lightweight.LastSeen = initial.LastSeen.Add(time.Second)
	store.Upsert(lightweight)

	updated, ok := store.Get(initial.NodeID)
	if !ok {
		t.Fatal("expected updated member")
	}
	if updated.Capabilities[cluster.CapabilityMemoryGB] != 8.0 {
		t.Fatalf("capabilities were not preserved: %+v", updated.Capabilities)
	}
	if updated.Metrics[cluster.MetricTemperatureC] != 42.0 {
		t.Fatalf("metrics were not preserved: %+v", updated.Metrics)
	}
	if len(updated.Engines) != 1 || updated.Engines[0].Engine != cluster.EngineLlamaCPP {
		t.Fatalf("engines were not preserved: %+v", updated.Engines)
	}
	if !updated.LastSeen.Equal(lightweight.LastSeen) {
		t.Fatalf("expected last_seen to advance, got %s", updated.LastSeen)
	}
}

func TestStorePreservesExistingRoleWhenIncomingRoleIsAuto(t *testing.T) {
	store := NewStore()

	first := memberWithRuntimeDetails()
	first.Role = NodeRoleJetson
	store.Upsert(first)

	second := first
	second.Role = NodeRoleAuto
	second.LastSeen = first.LastSeen.Add(time.Second)
	store.Upsert(second)

	updated, ok := store.Get(first.NodeID)
	if !ok {
		t.Fatal("expected updated member")
	}
	if updated.Role != NodeRoleJetson {
		t.Fatalf("expected preserved role %q, got %q", NodeRoleJetson, updated.Role)
	}
}

func TestStorePreservesLeaderPreferenceWhenIncomingPreferenceIsZero(t *testing.T) {
	store := NewStore()

	first := memberWithRuntimeDetails()
	first.LeaderPreference = 10
	store.Upsert(first)

	second := first
	second.LeaderPreference = 0
	second.LastSeen = first.LastSeen.Add(time.Second)
	store.Upsert(second)

	updated, ok := store.Get(first.NodeID)
	if !ok {
		t.Fatal("expected updated member")
	}
	if updated.LeaderPreference != 10 {
		t.Fatalf("expected preserved leader preference 10, got %d", updated.LeaderPreference)
	}
}

func TestStoreListReturnsMembersSortedByNodeID(t *testing.T) {
	store := NewStore()
	store.Upsert(storeTestMember("node-c"))
	store.Upsert(storeTestMember("node-a"))
	store.Upsert(storeTestMember("node-b"))

	members := store.List()
	if len(members) != 3 {
		t.Fatalf("expected 3 members, got %d", len(members))
	}

	got := []string{members[0].NodeID, members[1].NodeID, members[2].NodeID}
	want := []string{"node-a", "node-b", "node-c"}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("List() order = %+v, want %+v", got, want)
		}
	}
}

func TestStorePruneRemovesStaleMembersButKeepsSelf(t *testing.T) {
	store := NewStore()
	now := time.Unix(10, 0).UTC()

	self := storeTestMember("self")
	self.LastSeen = now.Add(-time.Hour)

	stalePeer := storeTestMember("stale")
	stalePeer.LastSeen = now.Add(-time.Hour)

	freshPeer := storeTestMember("fresh")
	freshPeer.LastSeen = now

	store.Upsert(self)
	store.Upsert(stalePeer)
	store.Upsert(freshPeer)

	removed := store.Prune(now, time.Minute, "self")
	if len(removed) != 1 || removed[0].NodeID != "stale" {
		t.Fatalf("expected stale peer removed, got %+v", removed)
	}
	if _, ok := store.Get("self"); !ok {
		t.Fatal("expected self to be kept")
	}
	if _, ok := store.Get("fresh"); !ok {
		t.Fatal("expected fresh peer to be kept")
	}
	if _, ok := store.Get("stale"); ok {
		t.Fatal("expected stale peer to be pruned")
	}
}

func storeTestMember(nodeID string) Member {
	return Member{
		ClusterID: "home-lab",
		NodeID:    nodeID,
		NodeName:  nodeID,
		Role:      NodeRoleJetson,
		APIURL:    "http://" + nodeID + ".local:52415",
		StartedAt: time.Unix(1, 0).UTC(),
		LastSeen:  time.Unix(2, 0).UTC(),
	}
}

func memberWithRuntimeDetails() Member {
	return Member{
		ClusterID: "home-lab",
		NodeID:    "node-1",
		NodeName:  "dopey",
		APIURL:    "http://dopey.local:52415",
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB: 8.0,
		},
		Metrics: map[string]any{
			cluster.MetricTemperatureC: 42.0,
		},
		Engines: []cluster.EngineEndpoint{{
			Engine:           cluster.EngineLlamaCPP,
			BaseURL:          "http://dopey.local:52415",
			OpenAICompatible: true,
		}},
		StartedAt: time.Unix(1, 0).UTC(),
		LastSeen:  time.Unix(2, 0).UTC(),
	}
}
