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
		if request.Method == http.MethodGet {
			if request.Header.Get(api.HeaderCoordinatorNodeID) != "" || request.Header.Get(api.HeaderClusterToken) != "" {
				t.Fatal("read-only status request included lifecycle credentials")
			}
		} else {
			if got := request.Header.Get(api.HeaderCoordinatorNodeID); got != "node-leader" {
				t.Fatalf("coordinator header=%q, want node-leader", got)
			}
			if got := request.Header.Get(api.HeaderClusterToken); got != "cluster-secret" {
				t.Fatalf("cluster token header=%q, want cluster-secret", got)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if request.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"activated":true,"resident":true,"active":true,"state":"active","deployment":{"deployment_id":"deployment-a","epoch":7,"model_id":"model-a","model_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"model_memory":{"layer_start":0,"layer_end":2,"layer_count":4,"resident_weight_bytes":180,"total_weight_bytes":400,"resident_tensor_count":12,"partitioned":true,"pinned":true}}`))
			return
		}
		_, _ = w.Write([]byte(`{"resident":true,"active":false,"state":"ready","deployment":{"deployment_id":"deployment-a","epoch":7,"model_id":"model-a","model_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"model_memory":{"layer_start":0,"layer_end":2,"layer_count":4,"resident_weight_bytes":180,"total_weight_bytes":400,"resident_tensor_count":12,"partitioned":true,"pinned":false}}`))
	}))
	defer server.Close()

	client := NewHTTPDeploymentClient(HTTPDeploymentClientConfig{
		Timeout:           time.Second,
		CoordinatorNodeID: "node-leader",
		ClusterToken:      "cluster-secret",
	})
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
	if status.Deployment == nil || status.Deployment.Epoch != 7 || status.Deployment.ModelSHA256 != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		t.Fatalf("unexpected deployment identity: %+v", status.Deployment)
	}
	identity := DeploymentIdentity{
		DeploymentID: "deployment-a",
		Epoch:        7,
		ModelID:      "model-a",
		ModelSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	activated, err := client.Activate(context.Background(), server.URL, identity)
	if err != nil {
		t.Fatalf("Activate() error = %v", err)
	}
	if !activated.Activated || !activated.Active || activated.Deployment == nil || *activated.Deployment != identity {
		t.Fatalf("unexpected activation response: %+v", activated)
	}
}
