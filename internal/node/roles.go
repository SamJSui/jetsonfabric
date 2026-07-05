package node

import (
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
	"github.com/SamJSui/jetsonfabric/internal/system"
)

func resolveNodeRole(role membership.NodeRole) membership.NodeRole {
	role = membership.NormalizeRole(role)
	if role != membership.NodeRoleAuto {
		return role
	}
	return inferNodeRole(system.Detect())
}

func inferNodeRole(snapshot system.Snapshot) membership.NodeRole {
	if snapshot.WSL {
		return membership.NodeRoleTest
	}
	if snapshotDeviceClass(snapshot) == cluster.DeviceClassJetson {
		return membership.NodeRoleJetson
	}
	return membership.NodeRoleWorker
}

func snapshotDeviceClass(snapshot system.Snapshot) cluster.DeviceClass {
	if len(snapshot.Capabilities) == 0 {
		return cluster.DeviceClassUnknown
	}
	value, ok := snapshot.Capabilities[cluster.CapabilityDeviceClass]
	if !ok {
		return cluster.DeviceClassUnknown
	}
	if typed, ok := value.(cluster.DeviceClass); ok {
		return typed
	}
	if typed, ok := value.(string); ok {
		return cluster.DeviceClass(typed)
	}
	return cluster.DeviceClassUnknown
}
