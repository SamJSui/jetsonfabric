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
		ClusterID:        "home-lab",
		NodeID:           "node-1234567890",
		NodeName:         "dopey",
		Hostname:         "dopey",
		Role:             membership.NodeRoleJetson,
		APIURL:           "http://dopey.local:52415",
		RuntimeURL:       "http://127.0.0.1:9090",
		LeaderPreference: 20,
		Arch:             "arm64",
		OS:               cluster.OperatingSystemLinux,
		StartedAt:        startedAt,
	}

	decoded, ok := memberFromMDNSTXT(memberTXT(member, 52415), "192.168.1.50")
	if !ok {
		t.Fatal("expected member TXT to decode")
	}
	assertDecodedIdentity(t, decoded, member)
	if decoded.APIURL != "http://192.168.1.50:52415" {
		t.Fatalf("expected source-ip API URL, got %s", decoded.APIURL)
	}
	if decoded.Role != membership.NodeRoleJetson || decoded.LeaderPreference != 20 {
		t.Fatalf("unexpected role metadata: %+v", decoded)
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
	member := mdnsTestMember()
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

func assertDecodedIdentity(t *testing.T, decoded membership.Member, expected membership.Member) {
	t.Helper()
	if decoded.ClusterID != expected.ClusterID || decoded.NodeID != expected.NodeID || decoded.NodeName != expected.NodeName {
		t.Fatalf("unexpected decoded identity: %+v", decoded)
	}
}

func mdnsTestMember() membership.Member {
	return membership.Member{
		ClusterID:        "home-lab",
		NodeID:           "node-abcdef",
		NodeName:         "dopey",
		Hostname:         "dopey",
		Role:             membership.NodeRoleJetson,
		APIURL:           "http://dopey.local:52415",
		LeaderPreference: 20,
		Arch:             "arm64",
		OS:               cluster.OperatingSystemLinux,
		StartedAt:        time.Now().UTC(),
	}
}
