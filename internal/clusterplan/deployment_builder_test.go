package clusterplan

import (
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestBuildDeploymentPlanUsesOneFreshCompatibleSnapshot(t *testing.T) {
	now := time.Date(2026, 7, 21, 0, 0, 0, 0, time.UTC)
	members := []membership.Member{
		deploymentMember("node-a", "dopey", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now),
		deploymentMember("node-b", "sleepy", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now),
	}
	result, err := BuildDeploymentPlan(DeploymentBuildRequest{
		Identity:   DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 3},
		Model:      deployableModel(nil),
		Members:    members,
		Now:        now,
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 2, AllowColocatedStages: true},
	})
	if err != nil {
		t.Fatalf("BuildDeploymentPlan() error = %v", err)
	}
	if result.Plan.Identity() != (DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 3}) {
		t.Fatalf("unexpected identity: %+v", result.Plan.Identity())
	}
	if result.Plan.Model().ModelSHA256 != strings.Repeat("a", 64) {
		t.Fatalf("unexpected model identity: %+v", result.Plan.Model())
	}
	if result.Plan.StageCount() != 2 || !result.Preview.Valid {
		t.Fatalf("unexpected route: %+v", result.Preview)
	}
	if result.Compatibility.Architecture != "arm64" || result.Compatibility.RuntimeRevision != "runtime-a" {
		t.Fatalf("unexpected compatibility: %+v", result.Compatibility)
	}

	members[0].NodeName = "mutated"
	if result.Plan.Stages()[0].NodeName != "dopey" {
		t.Fatal("plan changed after membership snapshot mutation")
	}
}

func TestBuildDeploymentPlanRejectsRevisionMismatch(t *testing.T) {
	now := time.Now().UTC()
	members := []membership.Member{
		deploymentMember("node-a", "dopey", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now),
		deploymentMember("node-b", "sleepy", "arm64", "runtime-b", "llama-a", cluster.ComputeBackendCPU, false, now),
	}
	_, err := BuildDeploymentPlan(DeploymentBuildRequest{
		Identity:   DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 1},
		Model:      deployableModel(nil),
		Members:    members,
		Now:        now,
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 2, AllowColocatedStages: true},
	})
	if err == nil || !strings.Contains(err.Error(), "matching architecture") {
		t.Fatalf("BuildDeploymentPlan() error = %v", err)
	}
}

func TestBuildDeploymentPlanRequiresActiveCUDACompatibility(t *testing.T) {
	now := time.Now().UTC()
	preferred := cluster.ComputeBackendCUDA
	members := []membership.Member{
		deploymentMember("node-a", "dopey", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCUDA, true, now),
		deploymentMember("node-b", "sleepy", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCUDA, false, now),
	}
	request := DeploymentBuildRequest{
		Identity:   DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 1},
		Model:      deployableModel(&preferred),
		Members:    members,
		Now:        now,
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 2, AllowColocatedStages: true},
	}
	if _, err := BuildDeploymentPlan(request); err == nil {
		t.Fatal("BuildDeploymentPlan() accepted a CUDA runtime without active CUDA attestation")
	}

	request.Members[1].Capabilities[cluster.CapabilityRuntimeCUDAActive] = true
	if _, err := BuildDeploymentPlan(request); err != nil {
		t.Fatalf("BuildDeploymentPlan() error after CUDA attestation = %v", err)
	}
}

func TestBuildDeploymentPlanExcludesStartupPinnedRuntimes(t *testing.T) {
	now := time.Now().UTC()
	pinned := deploymentMember("node-pinned", "pinned", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now)
	pinned.Capabilities[cluster.CapabilityRuntimeStartsIdle] = false
	pinned.Capabilities[cluster.CapabilityMemoryGB] = 256.0
	members := []membership.Member{
		pinned,
		deploymentMember("node-a", "dopey", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now),
		deploymentMember("node-b", "sleepy", "arm64", "runtime-a", "llama-a", cluster.ComputeBackendCPU, false, now),
	}
	members[1].Hostname = "host-a"
	members[2].Hostname = "host-b"

	result, err := BuildDeploymentPlan(DeploymentBuildRequest{
		Identity:   DeploymentIdentity{DeploymentID: "deployment-a", Epoch: 1},
		Model:      deployableModel(nil),
		Members:    members,
		Now:        now,
		StaleAfter: time.Minute,
		Policy:     Policy{StageCount: 2},
	})
	if err != nil {
		t.Fatalf("BuildDeploymentPlan() error = %v", err)
	}
	for _, stage := range result.Plan.Stages() {
		if stage.NodeID == pinned.NodeID {
			t.Fatalf("startup-pinned runtime was selected for managed deployment: %+v", stage)
		}
	}
}

func deployableModel(preferred *cluster.ComputeBackend) cluster.ModelProfile {
	return cluster.ModelProfile{
		ID:               "model-a",
		Family:           "llm",
		SupportedEngines: []cluster.Engine{cluster.EngineLlamaCPP},
		LayerCount:       28,
		PreferredCompute: preferred,
		PlacementModes:   []cluster.ExecutionMode{cluster.ExecutionModePipelineParallel},
		ArtifactPath:     "/models/model-a.gguf",
		ArtifactSHA256:   strings.Repeat("a", 64),
	}
}

func deploymentMember(
	nodeID string,
	nodeName string,
	arch string,
	runtimeRevision string,
	llamaRevision string,
	backend cluster.ComputeBackend,
	cudaActive bool,
	now time.Time,
) membership.Member {
	return membership.Member{
		ClusterID: "test-cluster",
		NodeID:    nodeID,
		NodeName:  nodeName,
		Hostname:  "test-host",
		APIURL:    "http://" + nodeID + ".test",
		Arch:      arch,
		Capabilities: map[string]any{
			cluster.CapabilityMemoryGB:                64.0,
			cluster.CapabilityComputeBackends:         []string{string(backend)},
			cluster.CapabilityRuntimeEngine:           string(cluster.EngineLlamaCPP),
			cluster.CapabilityRuntimeComputeBackend:   string(backend),
			cluster.CapabilityRuntimeExecutionMode:    string(cluster.ExecutionModePipelineParallel),
			cluster.CapabilityRuntimeRevision:         runtimeRevision,
			cluster.CapabilityRuntimeLlamaCPPRevision: llamaRevision,
			cluster.CapabilityRuntimeCUDAActive:       cudaActive,
			cluster.CapabilityRuntimeStartsIdle:       true,
		},
		StartedAt: now.Add(-time.Hour),
		LastSeen:  now,
	}
}
