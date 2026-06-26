package routing

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestPreviewRejectsMissingAccelerator(t *testing.T) {
	accel := cluster.AcceleratorCUDA
	model := cluster.ModelProfile{
		ID:                   "model",
		MinMemoryGB:          4,
		PreferredAccelerator: &accel,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeID: "cpu-node",
			Capabilities: map[string]any{
				cluster.CapabilityMemoryGB:     8.0,
				cluster.CapabilityAccelerators: []any{},
			},
		},
	}
	preview := Preview(model, nodes)
	placement := firstPlacement(t, preview)
	if placement.Valid {
		t.Fatalf("expected placement to be invalid")
	}
	expectedReason := MissingAcceleratorReason(cluster.AcceleratorCUDA)
	if placement.Reason != expectedReason {
		t.Fatalf("unexpected reason: %s", placement.Reason)
	}
}

func TestPreviewAcceptsCandidate(t *testing.T) {
	accel := cluster.AcceleratorCUDA
	model := cluster.ModelProfile{
		ID:                   "model",
		MinMemoryGB:          4,
		PreferredAccelerator: &accel,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeID: "jetson-node",
			Capabilities: map[string]any{
				cluster.CapabilityMemoryGB:     8.0,
				cluster.CapabilityAccelerators: []any{cluster.AcceleratorCUDA, cluster.AcceleratorJetson},
			},
		},
	}
	preview := Preview(model, nodes)
	placement := firstPlacement(t, preview)
	if !placement.Valid {
		t.Fatalf("expected placement to be valid: %+v", placement)
	}
}

func firstPlacement(t *testing.T, preview RoutePreview) PlacementPreview {
	t.Helper()
	for _, placement := range preview.Placements {
		return placement
	}
	t.Fatal("expected at least one placement")
	return PlacementPreview{}
}
