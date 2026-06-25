package routing

import (
	"testing"

	"github.com/SamJSui/JetsonMesh/internal/cluster"
)

func TestPreviewRejectsMissingAccelerator(t *testing.T) {
	accel := "cuda"
	model := cluster.ModelProfile{
		ID:                   "model",
		MinMemoryGB:          4,
		PreferredAccelerator: &accel,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeID: "cpu-node",
			Capabilities: map[string]any{
				"memory_gb":    8.0,
				"accelerators": []any{},
			},
		},
	}
	preview := Preview(model, nodes)
	if preview.Placements[0].Valid {
		t.Fatalf("expected placement to be invalid")
	}
	if preview.Placements[0].Reason != "missing_accelerator:cuda" {
		t.Fatalf("unexpected reason: %s", preview.Placements[0].Reason)
	}
}

func TestPreviewAcceptsCandidate(t *testing.T) {
	accel := "cuda"
	model := cluster.ModelProfile{
		ID:                   "model",
		MinMemoryGB:          4,
		PreferredAccelerator: &accel,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeID: "jetson-node",
			Capabilities: map[string]any{
				"memory_gb":    8.0,
				"accelerators": []any{"cuda", "jetson"},
			},
		},
	}
	preview := Preview(model, nodes)
	if !preview.Placements[0].Valid {
		t.Fatalf("expected placement to be valid: %+v", preview.Placements[0])
	}
}
