package routing

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestPreviewRejectsMissingCompute(t *testing.T) {
	compute := cluster.ComputeBackendCUDA
	model := cluster.ModelProfile{
		ID:               "model",
		MinMemoryGB:      4,
		PreferredCompute: &compute,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeName: "cpu-node",
			Capabilities: map[string]any{
				cluster.CapabilityMemoryGB:        8.0,
				cluster.CapabilityComputeBackends: []any{string(cluster.ComputeBackendCPU)},
			},
		},
	}
	preview := Preview(model, nodes)
	placement := firstPlacement(t, preview)
	if placement.Valid {
		t.Fatalf("expected placement to be invalid")
	}
	expectedReason := MissingComputeReason(string(cluster.ComputeBackendCUDA))
	if placement.Reason != expectedReason {
		t.Fatalf("unexpected reason: %s", placement.Reason)
	}
}

func TestPreviewAcceptsCandidate(t *testing.T) {
	compute := cluster.ComputeBackendCUDA
	model := cluster.ModelProfile{
		ID:               "model",
		MinMemoryGB:      4,
		PreferredCompute: &compute,
	}
	nodes := []cluster.NodeRecord{
		{
			NodeName: "jetson-node",
			Capabilities: map[string]any{
				cluster.CapabilityMemoryGB:        8.0,
				cluster.CapabilityDeviceClass:     string(cluster.DeviceClassJetson),
				cluster.CapabilityComputeBackends: []any{string(cluster.ComputeBackendCPU), string(cluster.ComputeBackendCUDA)},
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
