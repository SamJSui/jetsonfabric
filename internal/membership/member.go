package membership

import (
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

// NodeRole describes the semantic role a node plays in the fabric.
type NodeRole string

const (
	NodeRoleAuto        NodeRole = "auto"
	NodeRoleJetson      NodeRole = "jetson"
	NodeRoleCoordinator NodeRole = "coordinator"
	NodeRoleWorker      NodeRole = "worker"
	NodeRoleTest        NodeRole = "test"
)

// Member is a node's advertised view of itself within a JetsonFabric cluster.
//
// Membership is broader than the old control-plane node table. It exists on every
// jetsonfabric-node process and is used for discovery, leader election, facade
// proxying, and later deployment planning.
type Member struct {
	ClusterID string `json:"cluster_id"`

	NodeID   string `json:"node_id"`
	NodeName string `json:"node_name"`
	Hostname string `json:"hostname"`
	Role     NodeRole `json:"role,omitempty"`

	APIURL     string `json:"api_url"`
	RuntimeURL string `json:"runtime_url,omitempty"`

	LeaderPreference int  `json:"leader_preference,omitempty"`
	ControlEligible  bool `json:"control_eligible"`
	ControlPriority  int  `json:"control_priority"`

	Arch         string                  `json:"arch"`
	OS           cluster.OperatingSystem `json:"os"`
	Capabilities map[string]any          `json:"capabilities,omitempty"`
	Metrics      map[string]any          `json:"metrics,omitempty"`
	Engines      []cluster.EngineEndpoint `json:"engines,omitempty"`

	StartedAt time.Time `json:"started_at"`
	LastSeen  time.Time `json:"last_seen"`
}

func (m Member) Valid() bool {
	return strings.TrimSpace(m.ClusterID) != "" &&
		strings.TrimSpace(m.NodeID) != "" &&
		strings.TrimSpace(m.APIURL) != ""
}

func (m Member) IsStale(now time.Time, staleAfter time.Duration) bool {
	if staleAfter <= 0 {
		return false
	}
	if m.LastSeen.IsZero() {
		return true
	}
	return now.Sub(m.LastSeen) > staleAfter
}

func Normalize(m Member) Member {
	m.ClusterID = strings.TrimSpace(m.ClusterID)
	m.NodeID = strings.TrimSpace(m.NodeID)
	m.NodeName = strings.TrimSpace(m.NodeName)
	m.Hostname = strings.TrimSpace(m.Hostname)
	m.Role = NormalizeRole(m.Role)
	m.APIURL = strings.TrimSpace(m.APIURL)
	m.RuntimeURL = strings.TrimSpace(m.RuntimeURL)
	m.Arch = strings.TrimSpace(m.Arch)
	m.LeaderPreference = normalizeLeaderPreference(m)
	m.ControlPriority = m.LeaderPreference
	return m
}

func NormalizeRole(role NodeRole) NodeRole {
	value := strings.ToLower(strings.TrimSpace(string(role)))
	switch NodeRole(value) {
	case NodeRoleJetson, NodeRoleCoordinator, NodeRoleWorker, NodeRoleTest:
		return NodeRole(value)
	case NodeRoleAuto, "":
		return NodeRoleAuto
	default:
		return NodeRoleAuto
	}
}

func RoleLeaderEligible(role NodeRole) bool {
	switch NormalizeRole(role) {
	case NodeRoleCoordinator, NodeRoleJetson:
		return true
	default:
		return false
	}
}

func RoleRank(role NodeRole) int {
	switch NormalizeRole(role) {
	case NodeRoleCoordinator:
		return 100
	case NodeRoleJetson:
		return 80
	default:
		return 0
	}
}

func (m Member) EffectiveRole() NodeRole {
	role := NormalizeRole(m.Role)
	if role != NodeRoleAuto {
		return role
	}
	if memberDeviceClass(m) == cluster.DeviceClassJetson {
		return NodeRoleJetson
	}
	if m.ControlEligible {
		return NodeRoleJetson
	}
	return NodeRoleWorker
}

func (m Member) EffectiveLeaderPreference() int {
	return normalizeLeaderPreference(m)
}

func normalizeLeaderPreference(m Member) int {
	if m.LeaderPreference != 0 {
		return m.LeaderPreference
	}
	return m.ControlPriority
}

func memberDeviceClass(m Member) cluster.DeviceClass {
	if len(m.Capabilities) == 0 {
		return cluster.DeviceClassUnknown
	}
	value, ok := m.Capabilities[cluster.CapabilityDeviceClass]
	if !ok {
		return cluster.DeviceClassUnknown
	}
	return parseDeviceClass(value)
}

func parseDeviceClass(value any) cluster.DeviceClass {
	if typed, ok := value.(cluster.DeviceClass); ok {
		return typed
	}
	if typed, ok := value.(string); ok {
		return cluster.DeviceClass(strings.TrimSpace(typed))
	}
	return cluster.DeviceClassUnknown
}
