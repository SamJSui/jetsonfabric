package coordinator

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/runtimebridge"
)

func TestProbeDirectRuntimeEndpointsValidatesReadyDeployment(t *testing.T) {
	var status runtimebridge.DeploymentStatus
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/v1/deployment" {
			t.Fatalf("probe path = %q, want /v1/deployment", req.URL.Path)
		}
		writeRuntimeStatus(t, w, status)
	}))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	status = probeReadyRuntimeStatus(plan, plan.Stages()[0])
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err != nil {
		t.Fatalf("ready direct runtime was rejected: %v", err)
	}
}

func TestProbeDirectRuntimeEndpointsRejectsWrongDeployment(t *testing.T) {
	var status runtimebridge.DeploymentStatus
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeRuntimeStatus(t, w, status)
	}))
	defer server.Close()

	plan := directRuntimeProbePlan(t, server.URL)
	status = probeReadyRuntimeStatus(plan, plan.Stages()[0])
	status.Deployment.DeploymentID = "wrong-deployment"
	if err := probeDirectRuntimeEndpoints(t.Context(), plan); err == nil ||
		!strings.Contains(err.Error(), "wrong-deployment") {
		t.Fatalf("wrong runtime deployment error = %v", err)
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

func probeReadyRuntimeStatus(plan clusterplan.DeploymentPlan, stage clusterplan.Stage) runtimebridge.DeploymentStatus {
	return runtimebridge.DeploymentStatus{
		Resident: true,
		Active:   false,
		State:    "ready",
		Deployment: &runtimebridge.DeploymentIdentity{
			DeploymentID: plan.Identity().DeploymentID,
			Epoch:        plan.Identity().Epoch,
			ModelID:      plan.Model().ModelID,
			ModelSHA256:  plan.Model().ModelSHA256,
		},
		ModelMemory: &runtimebridge.ModelMemory{
			LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
			LayerCount:          plan.Model().LayerCount,
			ResidentWeightBytes: 100, TotalWeightBytes: 100,
			ResidentTensorCount: 1, Partitioned: false, Pinned: false,
		},
	}
}

func writeRuntimeStatus(t *testing.T, w http.ResponseWriter, status runtimebridge.DeploymentStatus) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(status); err != nil {
		t.Fatalf("encode runtime status: %v", err)
	}
}
