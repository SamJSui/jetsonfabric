package discovery

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestStaticSourceAnnouncesToSeeds(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	peer := testMember("peer", "peer", "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertAnnounceRequest(t, r, self.NodeID)
		_ = json.NewEncoder(w).Encode(AnnounceResponse{Members: []membership.Member{peer}})
	}))
	defer server.Close()

	source := NewStaticSource([]string{server.URL}, func() membership.Member { return self })
	members, err := source.Discover(t.Context())
	if err != nil {
		t.Fatalf("static discovery failed: %v", err)
	}
	if len(members) != 1 || members[0].NodeID != peer.NodeID {
		t.Fatalf("expected peer from seed, got %+v", members)
	}
}

func TestStaticSourceIgnoresEmptySeeds(t *testing.T) {
	source := NewStaticSource([]string{"", "   "}, func() membership.Member { return membership.Member{} })
	members, err := source.Discover(t.Context())
	if err != nil {
		t.Fatalf("empty seeds should not fail: %v", err)
	}
	if len(members) != 0 {
		t.Fatalf("expected no members, got %+v", members)
	}
}
