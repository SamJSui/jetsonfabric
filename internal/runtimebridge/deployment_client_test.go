package runtimebridge

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

func TestHTTPDeploymentClientSendsCoordinatorIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get(api.HeaderCoordinatorNodeID); got != "node-leader" {
			t.Fatalf("coordinator header=%q, want node-leader", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"resident":true,"active":false,"state":"ready","deployment":{"deployment_id":"deployment-a","model_id":"model-a"},"model_memory":{"layer_start":0,"layer_end":2,"layer_count":4,"resident_weight_bytes":180,"total_weight_bytes":400,"resident_tensor_count":12,"partitioned":true,"pinned":false}}`))
	}))
	defer server.Close()

	client := NewHTTPDeploymentClient(time.Second, "node-leader")
	status, err := client.Status(context.Background(), server.URL)
	if err != nil {
		t.Fatalf("Status() error = %v", err)
	}
	if !status.Resident || status.Active || status.State != "ready" || status.ModelMemory == nil {
		t.Fatalf("unexpected status: %+v", status)
	}
	if status.ModelMemory.LayerEnd != 2 || status.ModelMemory.ResidentWeightBytes != 180 || !status.ModelMemory.Partitioned {
		t.Fatalf("unexpected model memory: %+v", status.ModelMemory)
	}
}
