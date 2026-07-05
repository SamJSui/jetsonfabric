package discovery

import (
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestMDNSTXTMemberRoundTrip(t *testing.T) {
	startedAt := time.Date(2026, 7, 5, 4, 30, 0, 0, time.UTC)
	member := membership.Member{
		ClusterID:       "home-lab",
		NodeID:          "node-1234567890",
		NodeName:        "dopey",
		Hostname:        "dopey",
		APIURL:          "http://dopey.local:52415",
		RuntimeURL:      "http://127.0.0.1:9090",
		ControlEligible: true,
		ControlPriority: 20,
		Arch:            "arm64",
		OS:              cluster.OperatingSystemLinux,
		StartedAt:       startedAt,
	}

	decoded, ok := memberFromMDNSTXT(memberTXT(member, 52415), "192.168.1.50")
	if !ok {
		t.Fatal("expected member TXT to decode")
	}
	if decoded.ClusterID != member.ClusterID || decoded.NodeID != member.NodeID || decoded.NodeName != member.NodeName {
		t.Fatalf("unexpected decoded identity: %+v", decoded)
	}
	if decoded.APIURL != "http://192.168.1.50:52415" {
		t.Fatalf("expected source-ip API URL, got %s", decoded.APIURL)
	}
	if !decoded.ControlEligible || decoded.ControlPriority != 20 {
		t.Fatalf("unexpected control fields: %+v", decoded)
	}
}

func TestMDNSPacketAsksForService(t *testing.T) {
	cfg := MDNSConfig{Service: DefaultMDNSService, Domain: DefaultMDNSDomain}
	query := buildDNSQuery(serviceFQDN(cfg))
	if !packetAsksForService(query, serviceFQDN(cfg)) {
		t.Fatal("expected service query to match")
	}
	if packetAsksForService(query, "_other._tcp.local.") {
		t.Fatal("did not expect unrelated service to match")
	}
}

func TestParseTXTAnswers(t *testing.T) {
	member := membership.Member{
		ClusterID:       "home-lab",
		NodeID:          "node-abcdef",
		NodeName:        "dopey",
		Hostname:        "dopey",
		APIURL:          "http://dopey.local:52415",
		ControlEligible: true,
		ControlPriority: 20,
		Arch:            "arm64",
		OS:              cluster.OperatingSystemLinux,
		StartedAt:       time.Now().UTC(),
	}
	packet := buildDNSResponse(member, MDNSConfig{Service: DefaultMDNSService, Domain: DefaultMDNSDomain, Port: 52415})
	records := parseTXTAnswers(packet)
	if len(records) != 1 {
		t.Fatalf("expected one TXT record, got %d", len(records))
	}
	decoded, ok := memberFromMDNSTXT(records[0], "10.0.0.2")
	if !ok {
		t.Fatal("expected parsed TXT to decode to member")
	}
	if decoded.NodeID != member.NodeID || decoded.APIURL != "http://10.0.0.2:52415" {
		t.Fatalf("unexpected decoded member: %+v", decoded)
	}
}
