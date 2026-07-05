package facade

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
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

func decodeClusterView(t *testing.T, response *httptest.ResponseRecorder) ClusterView {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", response.Code)
	}
	var view ClusterView
	if err := json.NewDecoder(response.Body).Decode(&view); err != nil {
		t.Fatalf("decode cluster view: %v", err)
	}
	return view
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
