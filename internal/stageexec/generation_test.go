package stageexec

import (
	"context"
	"encoding/binary"
	"strings"
	"sync"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestGenerateDrivesDecodeLoopAndClosesSessions(t *testing.T) {
	deployment := stagewire.DeploymentIdentity{
		DeploymentID: "deployment-a",
		Epoch:        7,
		ModelSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	activation := make([]byte, 4*16*4)
	for index := range activation {
		activation[index] = byte((index * 23) % 251)
	}
	var mutex sync.Mutex
	closed := map[int]bool{}
	sessionID := ""

	checkSession := func(req StageRequest) {
		t.Helper()
		mutex.Lock()
		defer mutex.Unlock()
		if sessionID == "" {
			sessionID = req.SessionID
		}
		if req.SessionID == "" || req.SessionID != sessionID {
			t.Fatalf("session changed: got=%q want=%q", req.SessionID, sessionID)
		}
		if req.DeploymentIdentity != deployment {
			t.Fatalf("deployment identity changed: got=%+v want=%+v", req.DeploymentIdentity, deployment)
		}
	}

	stage0 := newFrameServer(t, func(req StageRequest) StageResponse {
		checkSession(req)
		if req.Operation == stagewire.OperationCloseSession {
			mutex.Lock()
			closed[0] = true
			mutex.Unlock()
			return closeSessionResponse(req)
		}
		if req.Operation != stagewire.OperationExecute {
			t.Fatalf("unexpected operation: %q", req.Operation)
		}
		if req.Phase == inference.PhasePrefill {
			if req.RequestID != "chatcmpl-1-prefill-0-stage-0" || req.DecodeStep != 0 || req.PayloadKind != stagewire.PayloadKindText {
				t.Fatalf("unexpected prefill request: %+v", req.Metadata)
			}
		} else if req.RequestID != "chatcmpl-1-decode-1-stage-0" || req.DecodeStep != 1 || req.PayloadKind != stagewire.PayloadKindSampledToken || binary.LittleEndian.Uint32(req.Payload) != 101 {
			t.Fatalf("unexpected decode request: %+v payload=%v", req.Metadata, req.Payload)
		}
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindActivation), Payload: activation}
	})
	defer stage0.Close()

	stage1 := newFrameServer(t, func(req StageRequest) StageResponse {
		checkSession(req)
		if req.Operation == stagewire.OperationCloseSession {
			mutex.Lock()
			closed[1] = true
			mutex.Unlock()
			return closeSessionResponse(req)
		}
		if !strings.HasSuffix(req.RequestID, "-stage-1") {
			t.Fatalf("stage 1 request ID=%q", req.RequestID)
		}
		payload := make([]byte, 4)
		metadata := responseMetadata(req, stagewire.PayloadKindSampledToken)
		metadata.CompletionTokens = 1
		if req.Phase == inference.PhasePrefill {
			binary.LittleEndian.PutUint32(payload, 101)
			metadata.Message = "A"
		} else {
			if req.DecodeStep != 1 {
				t.Fatalf("unexpected decode step: %d", req.DecodeStep)
			}
			binary.LittleEndian.PutUint32(payload, 102)
			metadata.Message = "B"
			metadata.CompletionTokens = 0
		}
		return StageResponse{Metadata: metadata, Payload: payload}
	})
	defer stage1.Close()

	result, err := New(Config{ClusterToken: testClusterToken}).Generate(context.Background(), Request{
		RequestID:  "chatcmpl-1",
		Model:      "model",
		Deployment: deployment,
		Payload:    "prompt",
		MaxTokens:  4,
		Plan:       testPlan(stage0.URL, stage1.URL),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if result.RequestID != "chatcmpl-1" || result.SessionID == "" || result.SessionID != sessionID {
		t.Fatalf("unexpected result IDs: %+v", result)
	}
	if result.GeneratedText != "A" || result.FinishReason != "stop" || !result.EndOfGeneration {
		t.Fatalf("unexpected generation result: %+v", result)
	}
	if len(result.SampledTokens) != 1 || result.SampledTokens[0] != 101 {
		t.Fatalf("sampled tokens: %v", result.SampledTokens)
	}
	if len(result.Stages) != 4 || result.Stages[2].Phase != inference.PhaseDecode || result.Stages[2].DecodeStep != 1 {
		t.Fatalf("unexpected traces: %+v", result.Stages)
	}
	mutex.Lock()
	defer mutex.Unlock()
	if !closed[0] || !closed[1] {
		t.Fatalf("sessions were not closed: %+v", closed)
	}
}

func closeSessionResponse(req StageRequest) StageResponse {
	metadata := responseMetadata(req, stagewire.PayloadKindText)
	return StageResponse{Metadata: metadata}
}

func TestCloseSessionRejectsMismatchedSuccessIdentity(t *testing.T) {
	server := newFrameServer(t, func(req StageRequest) StageResponse {
		response := closeSessionResponse(req)
		response.RequestID = "stale-cleanup-request"
		return response
	})
	defer server.Close()

	err := New(Config{ClusterToken: testClusterToken}).CloseSession(
		context.Background(),
		"session-a",
		"model",
		stagewire.DeploymentIdentity{
			DeploymentID: "deployment-a",
			Epoch:        7,
			ModelSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		testPlan(server.URL, server.URL),
	)
	if err == nil || !strings.Contains(err.Error(), "response identity") {
		t.Fatalf("unexpected cleanup result: %v", err)
	}
}
