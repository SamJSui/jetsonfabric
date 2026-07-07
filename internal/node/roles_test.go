package node

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

func TestInferNodeRoleFromJetsonDeviceClass(t *testing.T) {
	snapshot := snapshotWithDeviceClass(cluster.DeviceClassJetson)
	if role := inferNodeRole(snapshot); role != membership.NodeRoleJetson {
		t.Fatalf("expected jetson role, got %s", role)
	}
}

func TestInferNodeRoleFromLinuxPCDeviceClass(t *testing.T) {
	snapshot := snapshotWithDeviceClass(cluster.DeviceClassLinuxPC)
	if role := inferNodeRole(snapshot); role != membership.NodeRoleWorker {
		t.Fatalf("expected worker role, got %s", role)
	}
}

func snapshotWithDeviceClass(deviceClass cluster.DeviceClass) system.Snapshot {
	return system.Snapshot{
		Capabilities: map[string]any{
			cluster.CapabilityDeviceClass: string(deviceClass),
		},
	}
}
