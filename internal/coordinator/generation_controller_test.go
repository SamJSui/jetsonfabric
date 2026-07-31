package coordinator

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func TestGenerationControllerStartsStageZeroAndReleasesAdmission(t *testing.T) {
	generation := &recordingGenerationClient{}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: membershipMembersForRun{
			{nodeID: "node-a", apiURL: "http://node-a.test"},
			{nodeID: "node-b", apiURL: "http://node-b.test"},
		}.members()}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)

	session, err := server.generations.Start(context.Background(), generationSpec{
		ModelID:   "qwen2.5-coder-1.5b-q4",
		Prompt:    "prompt",
		MaxTokens: 4,
		Policy: clusterplan.Policy{
			StageCount:           2,
			AllowColocatedStages: true,
		},
	})
	if err != nil {
		t.Fatalf("start generation: %v", err)
	}
	if generation.calls != 1 || generation.nodeURL != "http://node-a.test" {
		t.Fatalf("generation did not start at stage zero: calls=%d url=%q", generation.calls, generation.nodeURL)
	}
	if len(generation.request.Stages) != 2 || generation.request.Stages[1].NodeID != "node-b" {
		t.Fatalf("runtime request did not carry the complete plan: %+v", generation.request.Stages)
	}
	if got := server.deployments.snapshot().InFlight; got != 1 {
		t.Fatalf("in-flight admissions=%d, want 1", got)
	}

	session.Close()
	if got := server.deployments.snapshot().InFlight; got != 0 {
		t.Fatalf("in-flight admissions after close=%d, want 0", got)
	}
}

func TestGenerationControllerReleasesAdmissionWhenRuntimeStartFails(t *testing.T) {
	generation := &recordingGenerationClient{err: errors.New("runtime unavailable")}
	server := NewServer(
		coordinatorTestRegistry(),
		WithMembershipSource(staticMemberSource{members: []membership.Member{
			membershipMembersForRun{{nodeID: "node-a", apiURL: "http://node-a.test"}}.members()[0],
		}}, time.Minute),
		WithClock(func() time.Time { return coordinatorTestNow() }),
		WithGenerationClient(generation),
	)

	_, err := server.generations.Start(context.Background(), generationSpec{
		ModelID:   "qwen2.5-coder-1.5b-q4",
		Prompt:    "prompt",
		MaxTokens: 1,
		Policy:    clusterplan.Policy{StageCount: 1},
	})
	var startError *generationStartError
	if !errors.As(err, &startError) || startError.kind != generationRuntimeUnavailable {
		t.Fatalf("unexpected start error: %v", err)
	}
	if got := server.deployments.snapshot().InFlight; got != 0 {
		t.Fatalf("failed runtime start leaked %d admissions", got)
	}
}

func TestGenerationControllerReportsUnknownEngineModel(t *testing.T) {
	server := NewServer(coordinatorTestRegistry())
	_, err := server.generations.Start(context.Background(), generationSpec{
		ModelID: "missing",
		Prompt:  "prompt",
	})
	var startError *generationStartError
	if !errors.As(err, &startError) || startError.kind != generationUnknownModel {
		t.Fatalf("unexpected unknown-model error: %v", err)
	}
}

func TestGenerationControllerIDsRemainUniqueWithFixedClock(t *testing.T) {
	controller := newGenerationController(
		coordinatorTestRegistry(),
		nil,
		time.Minute,
		func() time.Time { return coordinatorTestNow() },
		nil,
		nil,
	)

	const count = 256
	ids := make(chan string, count*2)
	var workers sync.WaitGroup
	for range count {
		workers.Add(1)
		go func() {
			defer workers.Done()
			requestID, sessionID := controller.nextGenerationIDs()
			ids <- requestID
			ids <- sessionID
		}()
	}
	workers.Wait()
	close(ids)

	seen := make(map[string]struct{}, count*2)
	for id := range ids {
		if _, exists := seen[id]; exists {
			t.Fatalf("duplicate generation ID %q", id)
		}
		seen[id] = struct{}{}
	}
	if len(seen) != count*2 {
		t.Fatalf("unique generation IDs=%d, want %d", len(seen), count*2)
	}
}
