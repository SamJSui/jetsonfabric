package membership

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestMemberValidRequiresClusterIDNodeIDAndAPIURL(t *testing.T) {
	validMember := validMemberTestMember()

	tests := []struct {
		name   string
		member Member
		valid  bool
	}{
		{
			name:   "valid member",
			member: validMember,
			valid:  true,
		},
		{
			name: "missing cluster id",
			member: func() Member {
				m := validMember
				m.ClusterID = ""
				return m
			}(),
			valid: false,
		},
		{
			name: "missing node id",
			member: func() Member {
				m := validMember
				m.NodeID = ""
				return m
			}(),
			valid: false,
		},
		{
			name: "missing api url",
			member: func() Member {
				m := validMember
				m.APIURL = ""
				return m
			}(),
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.member.Valid(); got != tt.valid {
				t.Fatalf("Valid() = %v, want %v", got, tt.valid)
			}
		})
	}
}

func TestMemberIsStale(t *testing.T) {
	now := memberTestNow()

	tests := []struct {
		name       string
		lastSeen   time.Time
		staleAfter time.Duration
		want       bool
	}{
		{"staleness disabled", now.Add(-time.Hour), 0, false},
		{"zero last seen", time.Time{}, time.Minute, true},
		{"fresh", now.Add(-30 * time.Second), time.Minute, false},
		{"exact boundary is not stale", now.Add(-time.Minute), time.Minute, false},
		{"past boundary is stale", now.Add(-time.Minute - time.Nanosecond), time.Minute, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			member := validMemberTestMember()
			member.LastSeen = tt.lastSeen

			if got := member.IsStale(now, tt.staleAfter); got != tt.want {
				t.Fatalf("IsStale() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeTrimsStringFields(t *testing.T) {
	member := Normalize(Member{
		ClusterID:  " home-lab ",
		NodeID:     " node-1 ",
		NodeName:   " dopey ",
		Hostname:   " dopey.local ",
		Role:       NodeRoleJetson,
		APIURL:     " http://dopey.local:52415 ",
		RuntimeURL: " http://127.0.0.1:9090 ",
		Arch:       " arm64 ",
	})

	if member.ClusterID != "home-lab" {
		t.Fatalf("ClusterID = %q", member.ClusterID)
	}
	if member.NodeID != "node-1" {
		t.Fatalf("NodeID = %q", member.NodeID)
	}
	if member.NodeName != "dopey" {
		t.Fatalf("NodeName = %q", member.NodeName)
	}
	if member.Hostname != "dopey.local" {
		t.Fatalf("Hostname = %q", member.Hostname)
	}
	if member.APIURL != "http://dopey.local:52415" {
		t.Fatalf("APIURL = %q", member.APIURL)
	}
	if member.RuntimeURL != "http://127.0.0.1:9090" {
		t.Fatalf("RuntimeURL = %q", member.RuntimeURL)
	}
	if member.Arch != "arm64" {
		t.Fatalf("Arch = %q", member.Arch)
	}
}

func TestNormalizeRoleDefaultsUnknownToAuto(t *testing.T) {
	tests := []struct {
		name string
		role NodeRole
		want NodeRole
	}{
		{"empty", "", NodeRoleAuto},
		{"unknown", "banana", NodeRoleAuto},
		{"auto", NodeRoleAuto, NodeRoleAuto},
		{"jetson", NodeRoleJetson, NodeRoleJetson},
		{"coordinator", NodeRoleCoordinator, NodeRoleCoordinator},
		{"worker", NodeRoleWorker, NodeRoleWorker},
		{"test", NodeRoleTest, NodeRoleTest},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NormalizeRole(tt.role); got != tt.want {
				t.Fatalf("NormalizeRole(%q) = %q, want %q", tt.role, got, tt.want)
			}
		})
	}
}

func TestEffectiveRoleUsesExplicitRole(t *testing.T) {
	member := validMemberTestMember()
	member.Role = NodeRoleCoordinator
	member.Capabilities = map[string]any{
		cluster.CapabilityDeviceClass: string(cluster.DeviceClassJetson),
	}

	if got := member.EffectiveRole(); got != NodeRoleCoordinator {
		t.Fatalf("EffectiveRole() = %q, want %q", got, NodeRoleCoordinator)
	}
}

func TestEffectiveRoleInfersJetsonFromDeviceClass(t *testing.T) {
	member := validMemberTestMember()
	member.Role = NodeRoleAuto
	member.Capabilities = map[string]any{
		cluster.CapabilityDeviceClass: string(cluster.DeviceClassJetson),
	}

	if got := member.EffectiveRole(); got != NodeRoleJetson {
		t.Fatalf("EffectiveRole() = %q, want %q", got, NodeRoleJetson)
	}
}

func TestEffectiveRoleDefaultsToWorkerForUnknownDevice(t *testing.T) {
	member := validMemberTestMember()
	member.Role = NodeRoleAuto
	member.Capabilities = nil

	if got := member.EffectiveRole(); got != NodeRoleWorker {
		t.Fatalf("EffectiveRole() = %q, want %q", got, NodeRoleWorker)
	}
}

func validMemberTestMember() Member {
	return Member{
		ClusterID: "home-lab",
		NodeID:    "node-1",
		NodeName:  "dopey",
		APIURL:    "http://dopey.local:52415",
		LastSeen:  memberTestNow(),
	}
}

func memberTestNow() time.Time {
	return time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC)
}
