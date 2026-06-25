package routing

import (
	"testing"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
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
	if preview.Placements[0].Valid {
		t.Fatalf("expected placement to be invalid")
	}
	expectedReason := MissingAcceleratorReason(cluster.AcceleratorCUDA)
	if preview.Placements[0].Reason != expectedReason {
		t.Fatalf("unexpected reason: %s", preview.Placements[0].Reason)
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
	if !preview.Placements[0].Valid {
		t.Fatalf("expected placement to be valid: %+v", preview.Placements[0])
	}
}
