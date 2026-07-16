package facade

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/election"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestClusterMembersHideStaleMembers(t *testing.T) {
	store := membership.NewStore()
	now := time.Now().UTC()
	store.Upsert(testFacadeMember("self", "dopey", membership.NodeRoleJetson, now))
	store.Upsert(testFacadeMember("stale", "wsl", membership.NodeRoleTest, now.Add(-time.Minute)))

	router := NewRouter(Config{SelfID: "self", Store: store, StaleAfter: 30 * time.Second})
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, PathClusterMembers, nil))

	view := decodeClusterView(t, response)
	if len(view.Members) != 1 || view.Members[0].NodeID != "self" {
		t.Fatalf("expected only fresh self member, got %+v", view.Members)
	}
}

func TestClusterElectionExplainsCandidates(t *testing.T) {
	store := membership.NewStore()
	now := time.Now().UTC()
	store.Upsert(testFacadeMember("self", "dopey", membership.NodeRoleJetson, now))
	store.Upsert(testFacadeMember("worker", "beehive", membership.NodeRoleWorker, now))

	router := NewRouter(Config{SelfID: "self", Store: store, StaleAfter: 30 * time.Second})
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, PathClusterElection, nil))

	result := decodeElectionResult(t, response)
	if result.Leader == nil || result.Leader.NodeID != "self" {
		t.Fatalf("unexpected leader: %+v", result.Leader)
	}
	if result.Epoch == 0 || result.LeaseExpiresAt == nil {
		t.Fatalf("expected lease-backed result, got %+v", result)
	}
	assertCandidateReason(t, result, "worker", election.ReasonRoleNotEligible)
}

func TestLayerSplitStageRoutesToLocalStageRunner(t *testing.T) {
	store := membership.NewStore()
	store.Upsert(testFacadeMember("self", "dopey", membership.NodeRoleJetson, time.Now().UTC()))

	stageCalled := false
	coordinatorCalled := false
	router := NewRouter(Config{
		SelfID:      "self",
		Store:       store,
		StaleAfter:  30 * time.Second,
		Coordinator: http.HandlerFunc(func(http.ResponseWriter, *http.Request) { coordinatorCalled = true }),
		StageRunner: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			stageCalled = true
			writeJSON(w, http.StatusAccepted, map[string]string{"status": "stage"})
		}),
	})

	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, api.PathLayerSplitStage, nil))

	if !stageCalled || coordinatorCalled {
		t.Fatalf("stageCalled=%v coordinatorCalled=%v", stageCalled, coordinatorCalled)
	}
	if response.Code != http.StatusAccepted {
		t.Fatalf("expected stage response, got %d", response.Code)
	}
}

func TestFollowerProxiesCoordinatorRequestsToLeader(t *testing.T) {
	leaderCalled := false
	leaderServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaderCalled = true
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected proxied path: %s", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, map[string]string{"served_by": "leader"})
	}))
	defer leaderServer.Close()

	store := membership.NewStore()
	now := time.Now().UTC()
	store.Upsert(testFacadeMember("self", "follower", membership.NodeRoleJetson, now))
	leader := testFacadeMember("leader", "leader", membership.NodeRoleCoordinator, now)
	leader.APIURL = leaderServer.URL
	store.Upsert(leader)

	router := NewRouter(Config{
		SelfID:     "self",
		Store:      store,
		StaleAfter: 30 * time.Second,
		Coordinator: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			t.Fatal("follower should not serve coordinator locally")
		}),
	})

	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if !leaderCalled {
		t.Fatal("expected leader server to be called")
	}
	if response.Code != http.StatusOK {
		t.Fatalf("expected OK, got %d: %s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "served_by", "leader")
}

func TestLeaderServesCoordinatorLocally(t *testing.T) {
	store := membership.NewStore()
	now := time.Now().UTC()
	store.Upsert(testFacadeMember("self", "leader", membership.NodeRoleCoordinator, now))

	localCalled := false
	router := NewRouter(Config{
		SelfID:     "self",
		Store:      store,
		StaleAfter: 30 * time.Second,
		Coordinator: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			localCalled = true
			if r.URL.Path != "/v1/models" {
				t.Fatalf("unexpected local coordinator path: %s", r.URL.Path)
			}
			writeJSON(w, http.StatusOK, map[string]string{"served_by": "local"})
		}),
	})

	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if !localCalled {
		t.Fatal("expected local coordinator to be called")
	}
	if response.Code != http.StatusOK {
		t.Fatalf("expected OK, got %d: %s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "served_by", "local")
}

func TestCoordinatorRequestReturnsLeaderUnavailableWhenNoLeader(t *testing.T) {
	store := membership.NewStore()
	now := time.Now().UTC()
	store.Upsert(testFacadeMember("worker", "worker", membership.NodeRoleWorker, now))

	router := NewRouter(Config{SelfID: "worker", Store: store, StaleAfter: 30 * time.Second})
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d: %s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "error", "leader_unavailable")
}

func TestAnnounceRejectsClusterMismatch(t *testing.T) {
	store := membership.NewStore()
	store.Upsert(testFacadeMember("self", "dopey", membership.NodeRoleJetson, time.Now().UTC()))

	router := NewRouter(Config{SelfID: "self", Store: store, StaleAfter: 30 * time.Second})
	body := strings.NewReader(`{
		"cluster_id": "other-cluster",
		"node_id": "peer",
		"node_name": "peer",
		"api_url": "http://peer.local:52415"
	}`)

	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, PathClusterAnnounce, body))

	if response.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "error", "cluster_mismatch")
}

func TestAnnounceUpsertsMemberAndReturnsClusterView(t *testing.T) {
	store := membership.NewStore()
	store.Upsert(testFacadeMember("self", "dopey", membership.NodeRoleJetson, time.Now().UTC()))

	router := NewRouter(Config{SelfID: "self", Store: store, StaleAfter: 30 * time.Second})
	body := strings.NewReader(`{
		"cluster_id": "home-lab",
		"node_id": "peer",
		"node_name": "peer",
		"role": "jetson",
		"api_url": "http://peer.local:52415"
	}`)

	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, PathClusterAnnounce, body))

	view := decodeClusterView(t, response)
	if _, ok := store.Get("peer"); !ok {
		t.Fatal("expected peer to be upserted")
	}
	if len(view.Members) != 2 {
		t.Fatalf("expected 2 members, got %+v", view.Members)
	}
}

func decodeClusterView(t *testing.T, response *httptest.ResponseRecorder) ClusterView {
	t.Helper()
	assertOK(t, response)
	var view ClusterView
	if err := json.NewDecoder(response.Body).Decode(&view); err != nil {
		t.Fatalf("decode cluster view: %v", err)
	}
	return view
}

func decodeElectionResult(t *testing.T, response *httptest.ResponseRecorder) election.Result {
	t.Helper()
	assertOK(t, response)
	var result election.Result
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode election result: %v", err)
	}
	return result
}

func assertOK(t *testing.T, response *httptest.ResponseRecorder) {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", response.Code)
	}
}

func assertCandidateReason(t *testing.T, result election.Result, nodeID string, reason string) {
	t.Helper()
	for _, candidate := range result.Candidates {
		if candidate.Member.NodeID == nodeID && candidate.Reason == reason {
			return
		}
	}
	t.Fatalf("expected %s reason %s in %+v", nodeID, reason, result.Candidates)
}

func assertJSONField(t *testing.T, body string, key string, want string) {
	t.Helper()
	var decoded map[string]string
	if err := json.Unmarshal([]byte(body), &decoded); err != nil {
		t.Fatalf("decode json: %v", err)
	}
	if decoded[key] != want {
		t.Fatalf("expected %s=%q, got %q in %s", key, want, decoded[key], body)
	}
}

func testFacadeMember(id string, name string, role membership.NodeRole, lastSeen time.Time) membership.Member {
	return membership.Member{
		ClusterID: "home-lab",
		NodeID:    id,
		NodeName:  name,
		Role:      role,
		APIURL:    "http://" + name + ".local:52415",
		StartedAt: lastSeen.Add(-time.Minute),
		LastSeen:  lastSeen,
	}
}
