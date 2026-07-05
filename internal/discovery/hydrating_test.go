package discovery

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestHydratingSourceUsesAnnounceResponse(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	fullPeer := testMember("peer", "peer", "")
	fullPeer.Capabilities = map[string]any{cluster.CapabilityMemoryGB: 8.0}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(AnnounceResponse{Members: []membership.Member{fullPeer}})
	}))
	defer server.Close()

	lightPeer := testMember("peer", "peer", server.URL)
	source := NewHydratingSource(staticMembers{lightPeer}, NewAnnounceClient(func() membership.Member { return self }))

	members, err := source.Discover(t.Context())
	if err != nil {
		t.Fatalf("discover failed: %v", err)
	}
	if len(members) != 1 {
		t.Fatalf("expected one member, got %d", len(members))
	}
	if members[0].Capabilities[cluster.CapabilityMemoryGB] != 8.0 {
		t.Fatalf("expected hydrated capabilities, got %+v", members[0].Capabilities)
	}
}

func TestHydratingSourceKeepsLightweightPeerOnFailure(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	peer := testMember("peer", "peer", "http://127.0.0.1:1")
	source := NewHydratingSource(staticMembers{peer}, NewAnnounceClient(func() membership.Member { return self }))

	members, err := source.Discover(t.Context())
	if err != nil {
		t.Fatalf("discover failed: %v", err)
	}
	if len(members) != 1 || members[0].NodeID != peer.NodeID {
		t.Fatalf("expected lightweight fallback, got %+v", members)
	}
}

type staticMembers []membership.Member

func (s staticMembers) Discover(context.Context) ([]membership.Member, error) {
	return []membership.Member(s), nil
}
