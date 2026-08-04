package coordinator

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
)

func TestProbeDirectRuntimeEndpointsValidatesHealth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/healthz" {
			t.Fatalf("probe path = %q, want /healthz", req.URL.Path)
		}
		_, _ = w.Write([]byte(`{"status":"ok","engine":"llama.cpp","stage_transport":"http_direct_v1"}`))
	}))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err != nil {
		t.Fatalf("healthy direct runtime was rejected: %v", err)
	}
}

func TestProbeDirectRuntimeEndpointsIgnoresPreferredDeployment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","engine":"llama.cpp","stage_transport":"http_direct_v1","model":"old-model"}`))
	}))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err != nil {
		t.Fatalf("active predecessor made direct runtime probe fail: %v", err)
	}
}

func TestProbeDirectRuntimeEndpointsRejectsWrongTransport(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","engine":"llama.cpp","stage_transport":"http_binary_v1"}`))
	}))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err == nil ||
		!strings.Contains(err.Error(), "http_binary_v1") {
		t.Fatalf("wrong runtime transport error = %v", err)
	}
}

func TestProbeDirectRuntimeEndpointsRejectsRedirect(t *testing.T) {
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Fatal("direct runtime probe followed a redirect")
	}))
	defer target.Close()
	server := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusTemporaryRedirect))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err == nil {
		t.Fatal("redirecting direct runtime endpoint was accepted")
	}
}

func directRuntimeProbePlan(t *testing.T, runtimeURL string) clusterplan.DeploymentPlan {
	t.Helper()
	plan, err := clusterplan.NewDeploymentPlan(clusterplan.DeploymentPlanSpec{
		Identity: clusterplan.DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 2},
		Model: clusterplan.DeploymentModelIdentity{
			ModelID: "model-a", ModelSHA256: strings.Repeat("a", 64),
			Engine: cluster.EngineLlamaCPP, ExecutionMode: cluster.ExecutionModePipelineParallel,
			StageTransport:     cluster.StageTransportHTTPDirectV1,
			ActivationEncoding: cluster.ActivationEncodingF16,
			KVCacheType:        cluster.KVCacheTypeF16,
			LayerCount:         28,
		},
		Stages: []clusterplan.Stage{{
			StageIndex: 0, StageCount: 1, NodeID: "node-a", NodeName: "dopey",
			PhysicalHostID: "dopey", APIURL: "http://dopey.local:52415",
			RuntimeURL: runtimeURL, LayerStart: 0, LayerEnd: 28,
		}},
	})
	if err != nil {
		t.Fatalf("build direct runtime probe plan: %v", err)
	}
	return plan
}
