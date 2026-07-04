package membership

import (
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
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

	APIURL     string `json:"api_url"`
	RuntimeURL string `json:"runtime_url,omitempty"`

	ControlEligible bool `json:"control_eligible"`
	ControlPriority int  `json:"control_priority"`

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
	m.APIURL = strings.TrimSpace(m.APIURL)
	m.RuntimeURL = strings.TrimSpace(m.RuntimeURL)
	m.Arch = strings.TrimSpace(m.Arch)
	return m
}
