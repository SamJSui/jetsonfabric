package stageexec

import (
	"context"
	"encoding/binary"
	"hash/crc32"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

func TestExecutorPassesBinaryPayloadBetweenStages(t *testing.T) {
	activation := make([]byte, 4*16*4)
	for i := range activation {
		activation[i] = byte((i * 19) % 251)
	}

	stage0 := newFrameServer(t, func(req StageRequest) StageResponse {
		if req.StageIndex != 0 || req.StageCount != 2 || string(req.Payload) != "prompt" {
			t.Fatalf("unexpected first request: %+v payload=%q", req.Metadata, req.Payload)
		}
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindActivation), Payload: activation}
	})
	defer stage0.Close()

	stage1 := newFrameServer(t, func(req StageRequest) StageResponse {
		if req.StageIndex != 1 || req.PayloadKind != stagewire.PayloadKindActivation {
			t.Fatalf("unexpected second request: %+v", req.Metadata)
		}
		if got, want := crc32.ChecksumIEEE(req.Payload), crc32.ChecksumIEEE(activation); got != want {
			t.Fatalf("activation crc=%08x want=%08x", got, want)
		}
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, crc32.ChecksumIEEE(req.Payload))
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindSampledToken), Payload: payload}
	})
	defer stage1.Close()

	result, err := New(Config{}).Execute(context.Background(), Request{
		RequestID: "request-1", Model: "model", Payload: "prompt",
		Plan: testPlan(stage0.URL, stage1.URL), StrictPayloadTransitions: true,
	})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if result.PayloadKind != stagewire.PayloadKindSampledToken || result.SampledToken == nil {
		t.Fatalf("unexpected result: %+v", result)
	}
	if len(result.Stages) != 2 {
		t.Fatalf("stage count=%d", len(result.Stages))
	}
	if result.Stages[0].PayloadKindOut != stagewire.PayloadKindActivation || result.Stages[1].PayloadKindIn != stagewire.PayloadKindActivation {
		t.Fatalf("unexpected traces: %+v", result.Stages)
	}
	if result.Stages[0].PayloadCRC32Out != result.Stages[1].PayloadCRC32In {
		t.Fatalf("activation checksum changed across handoff: %+v", result.Stages)
	}
}

func TestExecutorRetainsCompatibilityTextChainWhenStrictValidationDisabled(t *testing.T) {
	stage0 := newFrameServer(t, func(req StageRequest) StageResponse {
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindText), Payload: []byte("handoff")}
	})
	defer stage0.Close()
	stage1 := newFrameServer(t, func(req StageRequest) StageResponse {
		if string(req.Payload) != "handoff" {
			t.Fatalf("payload=%q", req.Payload)
		}
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindText), Payload: []byte("final")}
	})
	defer stage1.Close()

	result, err := New(Config{}).Execute(context.Background(), Request{
		RequestID: "request-1", Model: "model", Payload: "prompt", Plan: testPlan(stage0.URL, stage1.URL),
	})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if result.Payload != "final" || result.PayloadKind != stagewire.PayloadKindText {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestExecutorRejectsInvalidStrictTransition(t *testing.T) {
	stage0 := newFrameServer(t, func(req StageRequest) StageResponse {
		return StageResponse{Metadata: responseMetadata(req, stagewire.PayloadKindText), Payload: []byte("wrong")}
	})
	defer stage0.Close()
	stage1 := newFrameServer(t, func(req StageRequest) StageResponse {
		t.Fatal("second stage should not be called")
		return StageResponse{}
	})
	defer stage1.Close()

	_, err := New(Config{}).Execute(context.Background(), Request{
		RequestID: "request-1", Model: "model", Payload: "prompt",
		Plan: testPlan(stage0.URL, stage1.URL), StrictPayloadTransitions: true,
	})
	if err == nil {
		t.Fatal("expected payload contract error")
	}
}

func TestExecutorStopsAfterJSONFailure(t *testing.T) {
	stage0 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"runtime_down","message":"unavailable"}`))
	}))
	defer stage0.Close()
	stage1Called := false
	stage1 := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { stage1Called = true }))
	defer stage1.Close()

	result, err := New(Config{}).Execute(context.Background(), Request{
		RequestID: "request-1", Model: "model", Payload: "prompt", Plan: testPlan(stage0.URL, stage1.URL),
	})
	if err == nil || stage1Called {
		t.Fatalf("result=%+v err=%v called=%v", result, err, stage1Called)
	}
}

func newFrameServer(t *testing.T, run func(StageRequest) StageResponse) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.PathLayerSplitStage {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != stagewire.ContentType {
			t.Fatalf("content-type=%q", r.Header.Get("Content-Type"))
		}
		req, err := stagewire.Decode(r.Body)
		if err != nil {
			t.Fatalf("decode request: %v", err)
		}
		resp := run(req)
		encoded, err := stagewire.Marshal(resp)
		if err != nil {
			t.Fatalf("encode response: %v", err)
		}
		w.Header().Set("Content-Type", stagewire.ContentType)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(encoded)
	}))
}

func responseMetadata(req StageRequest, kind stagewire.PayloadKind) stagewire.Metadata {
	metadata := req.Metadata
	metadata.PayloadKind = kind
	metadata.BytesIn = int64(len(req.Payload))
	metadata.BytesOut = 0
	metadata.Encoding = ""
	metadata.DType = ""
	metadata.Shape = nil
	metadata.ByteOrder = ""
	metadata.Layout = ""
	if kind == stagewire.PayloadKindText {
		metadata.Encoding = "utf-8"
	}
	if kind == stagewire.PayloadKindActivation {
		metadata.DType = "f32"
		metadata.Shape = []int64{4, 16}
		metadata.ByteOrder = "little"
		metadata.Layout = "row_major"
	}
	if kind == stagewire.PayloadKindSampledToken {
		metadata.DType = "u32"
		metadata.Shape = []int64{1}
		metadata.ByteOrder = "little"
		metadata.Layout = "row_major"
	}
	return metadata
}

func testPlan(stage0URL, stage1URL string) clusterplan.RoutePreview {
	return clusterplan.RoutePreview{
		Model: "model", Valid: true, StageCount: 2,
		Stages: []clusterplan.Stage{
			{StageIndex: 0, StageCount: 2, NodeID: "node-a", NodeName: "node-a", APIURL: stage0URL, LayerStart: 0, LayerEnd: 1},
			{StageIndex: 1, StageCount: 2, NodeID: "node-b", NodeName: "node-b", APIURL: stage1URL, LayerStart: 1, LayerEnd: 2},
		},
	}
}

var _ = inference.PhasePrefill
