package coordinator

import (
	"strings"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

func TestSelectPipelineRuntimeMembersAllowsMixedComputeBackends(t *testing.T) {
	model, ok := coordinatorTestRegistry().Find("qwen2.5-coder-1.5b-q4")
	if !ok {
		t.Fatal("test model missing")
	}
	members := membershipMembersForRun{
		{nodeID: "node-a", apiURL: "http://node-a"},
		{nodeID: "node-b", apiURL: "http://node-b"},
	}.members()
	members[0].Capabilities[cluster.CapabilityRuntimeComputeBackend] = string(cluster.ComputeBackendCPU)
	members[1].Capabilities[cluster.CapabilityRuntimeComputeBackend] = string(cluster.ComputeBackendCUDA)

	selected, identity, err := selectPipelineRuntimeMembers(
		model,
		members,
		coordinatorTestNow(),
		time.Minute,
		2,
	)
	if err != nil {
		t.Fatalf("select mixed-backend pipeline: %v", err)
	}
	if len(selected) != 2 || identity.ModelSHA256 != coordinatorTestModelSHA256 {
		t.Fatalf("unexpected selection: members=%+v identity=%+v", selected, identity)
	}
}

func TestSelectPipelineRuntimeMembersRejectsDifferentArtifacts(t *testing.T) {
	model, ok := coordinatorTestRegistry().Find("qwen2.5-coder-1.5b-q4")
	if !ok {
		t.Fatal("test model missing")
	}
	members := membershipMembersForRun{
		{nodeID: "node-a", apiURL: "http://node-a", modelSHA256: strings.Repeat("a", 64)},
		{nodeID: "node-b", apiURL: "http://node-b", modelSHA256: strings.Repeat("b", 64)},
	}.members()

	if _, _, err := selectPipelineRuntimeMembers(model, members, coordinatorTestNow(), time.Minute, 2); err == nil {
		t.Fatal("expected different model artifacts to be rejected")
	}
}
