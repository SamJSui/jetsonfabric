package discovery

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterauth"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestAnnounceClientSendsSelfAndReturnsClusterView(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	peer := testMember("peer", "peer", "")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertAnnounceRequest(t, r, self.NodeID, "cluster-secret")
		writeAnnounceResponse(
			t,
			w,
			r,
			"cluster-secret",
			AnnounceResponse{Leader: &self, Members: []membership.Member{self, peer}},
		)
	}))
	defer server.Close()

	client := NewAnnounceClient(func() membership.Member { return self }, "cluster-secret")
	members, err := client.AnnounceURL(t.Context(), server.URL)
	if err != nil {
		t.Fatalf("announce failed: %v", err)
	}
	if len(members) != 3 {
		t.Fatalf("expected leader plus members, got %d", len(members))
	}
}

func TestAnnounceClientRejectsUnsignedOrMismatchedResponse(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	tests := []struct {
		name  string
		token string
	}{
		{name: "unsigned"},
		{name: "wrong token", token: "wrong-secret"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				writeAnnounceResponse(t, w, r, test.token, AnnounceResponse{Members: []membership.Member{self}})
			}))
			defer server.Close()

			client := NewAnnounceClient(func() membership.Member { return self }, "cluster-secret")
			if _, err := client.AnnounceURL(t.Context(), server.URL); err == nil {
				t.Fatal("unauthenticated announcement response was accepted")
			}
		})
	}
}

func TestAnnounceClientRejectsOversizedResponse(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(bytes.Repeat([]byte{'x'}, maxAnnouncementResponseSize+1))
	}))
	defer server.Close()

	client := NewAnnounceClient(func() membership.Member { return self }, "")
	if _, err := client.AnnounceURL(t.Context(), server.URL); err == nil {
		t.Fatal("oversized announcement response was accepted")
	}
}

func TestAnnounceClientSkipsSelfURL(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	client := NewAnnounceClient(func() membership.Member { return self }, "")

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

	client := NewAnnounceClient(func() membership.Member { return self }, "")
	if _, err := client.AnnounceURL(t.Context(), server.URL); err == nil {
		t.Fatal("expected announce failure")
	}
}

func TestAnnounceClientDoesNotFollowRedirects(t *testing.T) {
	self := testMember("self", "self", "http://self.local:52415")
	redirectFollowed := false
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirectFollowed = true
	}))
	defer target.Close()
	server := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusTemporaryRedirect))
	defer server.Close()

	client := NewAnnounceClient(func() membership.Member { return self }, "cluster-secret")
	if _, err := client.AnnounceURL(t.Context(), server.URL); err == nil {
		t.Fatal("redirecting announce endpoint was accepted")
	}
	if redirectFollowed {
		t.Fatal("announce client followed a redirect with authentication headers")
	}
}

func assertAnnounceRequest(t *testing.T, r *http.Request, wantNodeID string, wantToken string) {
	t.Helper()
	if r.URL.Path != pathClusterAnnounce {
		t.Fatalf("unexpected path: %s", r.URL.Path)
	}
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read announce: %v", err)
	}
	if wantToken != "" && !clusterauth.VerifyAnnouncementRequest(
		wantToken,
		r.Header.Get(clusterauth.HeaderAnnouncementTimestamp),
		r.Header.Get(clusterauth.HeaderAnnouncementNonce),
		r.Header.Get(clusterauth.HeaderAnnouncementSignature),
		payload,
		time.Now().UTC(),
	) {
		t.Fatal("announcement signature is invalid")
	}
	var member membership.Member
	if err := json.NewDecoder(bytes.NewReader(payload)).Decode(&member); err != nil {
		t.Fatalf("decode announce: %v", err)
	}
	if member.NodeID != wantNodeID {
		t.Fatalf("unexpected announced node: %s", member.NodeID)
	}
}

func writeAnnounceResponse(
	t *testing.T,
	w http.ResponseWriter,
	r *http.Request,
	token string,
	response AnnounceResponse,
) {
	t.Helper()
	payload, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("encode response: %v", err)
	}
	if token != "" {
		timestamp, signature := clusterauth.SignAnnouncementResponse(
			token,
			time.Now().UTC(),
			r.Header.Get(clusterauth.HeaderAnnouncementNonce),
			payload,
		)
		w.Header().Set(clusterauth.HeaderAnnouncementResponseTimestamp, timestamp)
		w.Header().Set(clusterauth.HeaderAnnouncementResponseSignature, signature)
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(payload)
}

func testMember(id string, name string, apiURL string) membership.Member {
	if apiURL == "" {
		apiURL = "http://" + name + ".local:52415"
	}
	return membership.Member{
		ClusterID: "home-lab",
		NodeID:    id,
		NodeName:  name,
		Hostname:  name,
		Role:      membership.NodeRoleJetson,
		APIURL:    apiURL,
		StartedAt: time.Unix(1, 0).UTC(),
		LastSeen:  time.Unix(2, 0).UTC(),
	}
}
