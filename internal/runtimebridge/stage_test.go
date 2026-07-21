package runtimebridge

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestStageProxyStreamsBinaryFrameWithoutChangingContentType(t *testing.T) {
	payload := bytes.Repeat([]byte{0, 1, 2, 3, 0xff}, 4096)
	frame := stagewire.Frame{
		Metadata: stagewire.Metadata{
			SessionID: "s", RequestID: "r", ModelID: "m",
			StageIndex: 1, StageCount: 2, NodeName: "node",
			LayerStart: 1, LayerEnd: 2, PayloadKind: stagewire.PayloadKindActivation,
			DType: "u8", Shape: []int64{int64(len(payload))},
		},
		Payload: payload,
	}
	encoded, err := stagewire.Marshal(frame)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected runtime path: %s", r.URL.Path)
		}
		if r.Header.Get(api.HeaderClusterToken) != "" || r.Header.Get(api.HeaderCoordinatorNodeID) != "" {
			t.Fatalf("fabric credentials reached runtime: token=%q coordinator=%q", r.Header.Get(api.HeaderClusterToken), r.Header.Get(api.HeaderCoordinatorNodeID))
		}
		if r.Header.Get("Content-Type") != stagewire.ContentType {
			t.Fatalf("content-type=%q", r.Header.Get("Content-Type"))
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(body, encoded) {
			t.Fatal("proxy changed binary request body")
		}
		w.Header().Set("Content-Type", stagewire.ContentType)
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write(encoded)
	}))
	defer runtime.Close()

	proxy, err := NewStageProxy(runtime.URL)
	if err != nil {
		t.Fatalf("create proxy: %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathLayerSplitStage, bytes.NewReader(encoded))
	request.Header.Set("Content-Type", stagewire.ContentType)
	request.Header.Set(api.HeaderClusterToken, "cluster-secret")
	request.Header.Set(api.HeaderCoordinatorNodeID, "coordinator-a")
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted || response.Header().Get("Content-Type") != stagewire.ContentType {
		t.Fatalf("unexpected response: %d %q", response.Code, response.Header().Get("Content-Type"))
	}
	if !bytes.Equal(response.Body.Bytes(), encoded) {
		t.Fatal("proxy changed binary response body")
	}
}

func TestStageProxyRejectsEmptyBody(t *testing.T) {
	proxy, err := NewStageProxy("http://127.0.0.1:9090")
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodPost, api.PathLayerSplitStage, nil))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", response.Code)
	}
	assertJSONField(t, response.Body.String(), "error", "stage_body_required")
}

func TestStageProxyRejectsNonPost(t *testing.T) {
	proxy, err := NewStageProxy("http://127.0.0.1:9090")
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodGet, api.PathLayerSplitStage, nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status=%d", response.Code)
	}
}

func TestStageProxyReturnsBadGatewayWhenRuntimeUnavailable(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	runtimeURL := "http://" + listener.Addr().String()
	_ = listener.Close()

	proxy, err := NewStageProxy(runtimeURL)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, httptest.NewRequest(http.MethodPost, api.PathLayerSplitStage, bytes.NewReader([]byte{1})))
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
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
