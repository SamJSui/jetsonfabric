package layersplit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

func TestNewTransportSelectsHTTPByDefault(t *testing.T) {
	transport, err := NewTransport("")
	if err != nil {
		t.Fatalf("new transport: %v", err)
	}
	if _, ok := transport.(*HTTPTransport); !ok {
		t.Fatalf("expected HTTP transport, got %T", transport)
	}
}

func TestNewTransportRejectsUnimplementedKinds(t *testing.T) {
	for _, kind := range []TransportKind{TransportGRPC, TransportCustomTCP} {
		t.Run(string(kind), func(t *testing.T) {
			if _, err := NewTransport(kind); err == nil {
				t.Fatalf("expected %s to be unavailable", kind)
			}
		})
	}
}

func TestHTTPTransportRunsStage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		var req ActivationRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		resp := SyntheticStageResponse(req, "agent-http", 0)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	transport := NewHTTPTransport(0)
	stage := Stage{Index: 1, NodeName: "agent-http", Role: StageRoleLast, LayerStart: 14, LayerEnd: 28}
	req := BuildStageRequest("session", "request", "model", stage, "prompt", TransportHTTP)
	resp, err := transport.RunStage(context.Background(), StageTarget{NodeName: "agent-http", BaseURL: server.URL, Stage: stage}, req)
	if err != nil {
		t.Fatalf("run stage: %v", err)
	}
	if resp.NodeName != "agent-http" || resp.LayerStart != 14 || resp.LayerEnd != 28 {
		t.Fatalf("unexpected response metadata: %+v", resp)
	}
	if resp.Payload != "prompt -> agent-http[14:28]" {
		t.Fatalf("unexpected payload: %s", resp.Payload)
	}
}

func TestLocalTransportRecordsRequests(t *testing.T) {
	transport := &LocalTransport{}
	stage := Stage{Index: 0, NodeName: "agent-local", Role: StageRoleFirst, LayerStart: 0, LayerEnd: 14}
	req := BuildStageRequest("session", "request", "model", stage, "prompt", TransportLocal)
	resp, err := transport.RunStage(context.Background(), StageTarget{NodeName: "agent-local", Stage: stage}, req)
	if err != nil {
		t.Fatalf("run stage: %v", err)
	}
	if len(transport.Requests) != 1 {
		t.Fatalf("expected one recorded request, got %d", len(transport.Requests))
	}
	if resp.Payload != "prompt -> agent-local[0:14]" {
		t.Fatalf("unexpected payload: %s", resp.Payload)
	}
}
