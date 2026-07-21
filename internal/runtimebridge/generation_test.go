package runtimebridge

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

func TestHTTPGenerationClientStartsAuthenticatedRuntimeStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != api.PathRuntimeGeneration {
			t.Fatalf("unexpected request: %s %s", request.Method, request.URL.Path)
		}
		if request.Header.Get(api.HeaderCoordinatorNodeID) != "leader-a" ||
			request.Header.Get(api.HeaderClusterToken) != "cluster-secret" {
			t.Fatalf("missing generation authentication headers: %v", request.Header)
		}
		var decoded GenerationRequest
		if err := json.NewDecoder(request.Body).Decode(&decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.RequestID != "request-a" || len(decoded.Stages) != 1 {
			t.Fatalf("unexpected generation request: %+v", decoded)
		}
		w.Header().Set("Content-Type", GenerationContentType)
		_, _ = io.WriteString(w, "{\"type\":\"done\",\"finish_reason\":\"length\",\"sampled_tokens\":[]}\n")
	}))
	defer server.Close()

	client := NewHTTPGenerationClient(HTTPGenerationClientConfig{
		CoordinatorNodeID: "leader-a",
		ClusterToken:      "cluster-secret",
	})
	stream, err := client.Start(context.Background(), server.URL, GenerationRequest{
		RequestID: "request-a", SessionID: "session-a", ModelID: "model-a", Prompt: "hello", MaxTokens: 1,
		Stages: []clusterplan.Stage{{StageIndex: 0, StageCount: 1, NodeID: "node-a", NodeName: "dopey", APIURL: server.URL, LayerEnd: 8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	payload, err := io.ReadAll(stream.Body)
	if err != nil || !strings.Contains(string(payload), `"type":"done"`) {
		t.Fatalf("unexpected stream: payload=%q err=%v", payload, err)
	}
}

func TestGenerationProxyStripsClusterCredentials(t *testing.T) {
	var received bool
	runtimeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		received = true
		if request.URL.Path != runtimeGenerationPath {
			t.Fatalf("runtime path=%q", request.URL.Path)
		}
		if request.Header.Get(api.HeaderCoordinatorNodeID) != "" || request.Header.Get(api.HeaderClusterToken) != "" {
			t.Fatalf("proxy leaked cluster credentials: %v", request.Header)
		}
		w.Header().Set("Content-Type", GenerationContentType)
		_, _ = io.WriteString(w, "{\"type\":\"token\",\"token\":7,\"index\":0}\n")
	}))
	defer runtimeServer.Close()
	proxy, err := NewGenerationProxy(runtimeServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, api.PathRuntimeGeneration, strings.NewReader(`{"request_id":"request-a"}`))
	request.Header.Set(api.HeaderCoordinatorNodeID, "leader-a")
	request.Header.Set(api.HeaderClusterToken, "cluster-secret")
	response := httptest.NewRecorder()
	proxy.ServeHTTP(response, request)
	if !received || response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"type":"token"`) {
		t.Fatalf("proxy response status=%d received=%v body=%s", response.Code, received, response.Body.String())
	}
}

func TestHTTPGenerationClientRejectsUnexpectedContentType(t *testing.T) {
	for _, contentType := range []string{"application/json", GenerationContentType + "-garbage"} {
		t.Run(contentType, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", contentType)
				_, _ = io.WriteString(w, `{}`)
			}))
			defer server.Close()
			client := NewHTTPGenerationClient(HTTPGenerationClientConfig{})
			_, err := client.Start(context.Background(), server.URL, GenerationRequest{})
			if err == nil || !strings.Contains(err.Error(), "content-type") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
