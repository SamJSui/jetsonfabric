package stageexec

import (
	"context"
	"encoding/binary"
	"sync"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestGenerateDrivesDecodeLoopAndClosesSessions(t *testing.T) {
	activation := make([]byte, 4*16*4)
	for index := range activation {
		activation[index] = byte((index * 23) % 251)
	}
	var mutex sync.Mutex
	closed := map[int]bool{}

	stage0 := newFrameServer(t, func(req StageRequest) StageResponse {
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
			if req.DecodeStep != 0 || req.PayloadKind != stagewire.PayloadKindText {
				t.Fatalf("unexpected prefill request: %+v", req.Metadata)
			}
		} else if req.DecodeStep != 1 || req.PayloadKind != stagewire.PayloadKindSampledToken || binary.LittleEndian.Uint32(req.Payload) != 101 {
			t.Fatalf("unexpected decode request: %+v payload=%v", req.Metadata, req.Payload)
		}
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindActivation), Payload: activation}
	})
	defer stage0.Close()

	stage1 := newFrameServer(t, func(req StageRequest) StageResponse {
		if req.Operation == stagewire.OperationCloseSession {
			mutex.Lock()
			closed[1] = true
			mutex.Unlock()
			return closeSessionResponse(req)
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

	result, err := New(Config{}).Generate(context.Background(), Request{
		RequestID: "generation-1",
		Model:     "model",
		Payload:   "prompt",
		MaxTokens: 4,
		Plan:      testPlan(stage0.URL, stage1.URL),
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if result.GeneratedText != "AB" || result.FinishReason != "stop" || !result.EndOfGeneration {
		t.Fatalf("unexpected generation result: %+v", result)
	}
	if len(result.SampledTokens) != 2 || result.SampledTokens[0] != 101 || result.SampledTokens[1] != 102 {
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
