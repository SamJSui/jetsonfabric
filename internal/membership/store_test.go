package membership

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

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
	if len(updated.Engines) != 1 || updated.Engines[0].Engine != cluster.EngineJetsonFabric {
		t.Fatalf("engines were not preserved: %+v", updated.Engines)
	}
	if !updated.LastSeen.Equal(lightweight.LastSeen) {
		t.Fatalf("expected last_seen to advance, got %s", updated.LastSeen)
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
			Engine:           cluster.EngineJetsonFabric,
			BaseURL:          "http://dopey.local:52415",
			OpenAICompatible: true,
		}},
		StartedAt: time.Unix(1, 0).UTC(),
		LastSeen:  time.Unix(2, 0).UTC(),
	}
}
