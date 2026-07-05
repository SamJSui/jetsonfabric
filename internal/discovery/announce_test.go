package discovery

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestAnnounceClientSendsSelfAndReturnsClusterView(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	peer := testMember("peer", "peer", "")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertAnnounceRequest(t, r, self.NodeID)
		writeAnnounceResponse(t, w, AnnounceResponse{Leader: &self, Members: []membership.Member{self, peer}})
	}))
	defer server.Close()

	client := NewAnnounceClient(func() membership.Member { return self })
	members, err := client.AnnounceURL(t.Context(), server.URL)
	if err != nil {
		t.Fatalf("announce failed: %v", err)
	}
	if len(members) != 3 {
		t.Fatalf("expected leader plus members, got %d", len(members))
	}
}

func TestAnnounceClientSkipsSelfURL(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	client := NewAnnounceClient(func() membership.Member { return self })

	members, err := client.AnnounceURL(t.Context(), "http://self.local:52415/")
	if err != nil {
		t.Fatalf("self announce should not fail: %v", err)
	}
	if len(members) != 0 {
		t.Fatalf("expected no self announce members, got %d", len(members))
	}
}

func TestAnnounceClientReturnsHTTPError(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	defer server.Close()

	client := NewAnnounceClient(func() membership.Member { return self })
	if _, err := client.AnnounceURL(t.Context(), server.URL); err == nil {
		t.Fatal("expected announce failure")
	}
}

func assertAnnounceRequest(t *testing.T, r *http.Request, wantNodeID string) {
	t.Helper()
	if r.URL.Path != pathClusterAnnounce {
		t.Fatalf("unexpected path: %s", r.URL.Path)
	}
	var member membership.Member
	if err := json.NewDecoder(r.Body).Decode(&member); err != nil {
		t.Fatalf("decode announce: %v", err)
	}
	if member.NodeID != wantNodeID {
		t.Fatalf("unexpected announced node: %s", member.NodeID)
	}
}

func writeAnnounceResponse(t *testing.T, w http.ResponseWriter, response AnnounceResponse) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}

func testMember(id string, name string, apiURL string) membership.Member {
	if apiURL == "" {
		apiURL = "http://" + name + ".local:52415"
	}
	return membership.Member{
		ClusterID:  "home-lab",
		NodeID:     id,
		NodeName:   name,
		Hostname:   name,
		Role:       membership.NodeRoleJetson,
		APIURL:     apiURL,
		StartedAt:  time.Unix(1, 0).UTC(),
		LastSeen:   time.Unix(2, 0).UTC(),
	}
}
