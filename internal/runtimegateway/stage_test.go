package runtimegateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

func TestStageProxyForwardsRequestToRuntime(t *testing.T) {
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected runtime path: %s", r.URL.Path)
		}
		if r.Header.Get("X-Test") != "yes" {
			t.Fatalf("expected copied header")
		}
		writeJSON(w, http.StatusAccepted, map[string]string{"status": "ok"})
	}))
	defer runtime.Close()

	proxy, err := NewStageProxy(runtime.URL)
	if err != nil {
		t.Fatalf("create proxy: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitStage, strings.NewReader(`{"payload":"hello"}`))
	request.Header.Set("X-Test", "yes")
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)

	if response.Code != http.StatusAccepted {
		t.Fatalf("expected runtime status, got %d: %s", response.Code, response.Body.String())
	}
	assertJSONField(t, response.Body.String(), "status", "ok")
}

func TestStageProxyRejectsNonPost(t *testing.T) {
	proxy, err := NewStageProxy("http://127.0.0.1:9090")
	if err != nil {
		t.Fatalf("create proxy: %v", err)
	}

	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodGet, api.PathLayerSplitStage, nil))

	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected method rejection, got %d", response.Code)
	}
	assertJSONField(t, response.Body.String(), "error", "method_not_allowed")
}

func TestNewStageProxyRequiresAbsoluteRuntimeURL(t *testing.T) {
	if _, err := NewStageProxy("127.0.0.1:9090"); err == nil {
		t.Fatal("expected invalid URL error")
	}
}

func assertJSONField(t *testing.T, body string, key string, want string) {
	t.Helper()
	var decoded map[string]string
	if err := json.Unmarshal([]byte(body), &decoded); err != nil {
		t.Fatalf("decode json: %v", err)
	}
	if decoded[key] != want {
		t.Fatalf("expected %s=%q, got %q", key, want, decoded[key])
	}
}
