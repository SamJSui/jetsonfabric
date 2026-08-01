package stagewire

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/inference"
)

func TestActivationFrameRoundTripPreservesBinaryPayload(t *testing.T) {
	payload := make([]byte, 4*16*4)
	for i := range payload {
		payload[i] = byte((i * 37) % 251)
	}
	payload[3] = 0
	payload[127] = 0

	encoded, err := Marshal(Frame{
		Metadata: Metadata{
			SessionID: "session-1", RequestID: "request-1", ModelID: "model",
			Phase: inference.PhasePrefill, StageIndex: 1, StageCount: 2,
			NodeName: "stage-1", LayerStart: 14, LayerEnd: 28,
			PayloadKind: PayloadKindActivation, DType: "f32", Shape: []int64{4, 16},
			ExecutionUS: 1200, ActivationDecodeUS: 10, ActivationEncodeUS: 20,
			StageTotalUS: 1300,
		},
		Payload: payload,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(encoded[HeaderSize:], []byte("AA==")) {
		t.Fatal("binary payload appears to be base64 encoded")
	}

	decoded, err := Unmarshal(encoded)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !bytes.Equal(decoded.Payload, payload) {
		t.Fatal("payload changed during round trip")
	}
	if decoded.PayloadCRC32 != crc32.ChecksumIEEE(payload) {
		t.Fatalf("crc32=%08x", decoded.PayloadCRC32)
	}
	if decoded.PayloadBytes != int64(len(payload)) || decoded.DType != "f32" || len(decoded.Shape) != 2 {
		t.Fatalf("unexpected metadata: %+v", decoded.Metadata)
	}
	if decoded.ExecutionUS != 1200 || decoded.ActivationDecodeUS != 10 ||
		decoded.ActivationEncodeUS != 20 || decoded.StageTotalUS != 1300 {
		t.Fatalf("timings changed during round trip: %+v", decoded.Metadata)
	}
}

func TestTextFrameRoundTrip(t *testing.T) {
	identity := DeploymentIdentity{
		DeploymentID: "deployment-a",
		Epoch:        3,
		ModelSHA256:  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}
	frame := Frame{
		Metadata: Metadata{
			SessionID: "s", RequestID: "r", ModelID: "m",
			DeploymentIdentity: identity,
			StageIndex:         0, StageCount: 1, NodeName: "node",
			LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindText,
		},
		Payload: []byte("hello"),
	}
	encoded, err := Marshal(frame)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := Unmarshal(encoded)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(decoded.Payload) != "hello" || decoded.Encoding != "utf-8" || decoded.Phase != inference.PhasePrefill || decoded.DeploymentIdentity != identity {
		t.Fatalf("unexpected frame: %+v", decoded)
	}
}

func TestVersionTwoMetadataRoundTrip(t *testing.T) {
	frame := Frame{
		Metadata: Metadata{
			Operation:          OperationExecute,
			SessionID:          "session-v2",
			RequestID:          "request-v2",
			ModelID:            "model-v2",
			Phase:              inference.PhaseDecode,
			DecodeStep:         7,
			StageIndex:         1,
			StageCount:         2,
			NodeName:           "stage-1",
			LayerStart:         14,
			LayerEnd:           28,
			PayloadKind:        PayloadKindSampledTokens,
			DType:              "u32",
			Shape:              []int64{2},
			MaxTokens:          128,
			BytesIn:            24576,
			BytesOut:           8,
			PromptTokens:       4,
			PromptTokenIDs:     []uint32{1, 2, 3, 4},
			CompletionTokens:   2,
			ExecutionBatchSize: 1,
			VerificationWidth:  2,
			LatencyMS:          12,
			ExecutionUS:        9000,
			ActivationDecodeUS: 100,
			ActivationEncodeUS: 200,
			StageTotalUS:       9500,
			TokenTextOffsets:   []uint32{0, 3, 8},
			TokenEOG:           []uint32{0, 1},
			MessageBytes:       []int{111, 107},
		},
		Payload: []byte{11, 0, 0, 0, 12, 0, 0, 0},
	}

	encoded, err := Marshal(frame)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := Unmarshal(encoded)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.ProtocolVersion != Version || decoded.Operation != OperationExecute ||
		decoded.VerificationWidth != 2 || decoded.ExecutionBatchSize != 1 ||
		!equalUint32s(decoded.PromptTokenIDs, frame.PromptTokenIDs) ||
		!equalUint32s(decoded.TokenTextOffsets, frame.TokenTextOffsets) ||
		!equalUint32s(decoded.TokenEOG, frame.TokenEOG) ||
		!bytes.Equal(decoded.Payload, frame.Payload) {
		t.Fatalf("version two metadata changed during round trip: %+v", decoded.Metadata)
	}
}

func TestRollbackFrameRequiresPositiveTokenCount(t *testing.T) {
	frame := Frame{Metadata: Metadata{
		Operation: OperationRollback,
		SessionID: "s", RequestID: "r", ModelID: "m",
		StageIndex: 0, StageCount: 1, NodeName: "node",
		LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindText,
	}}
	if _, err := Marshal(frame); err == nil {
		t.Fatal("expected zero-token rollback to be rejected")
	}
	frame.RollbackTokens = 2
	if _, err := Marshal(frame); err != nil {
		t.Fatalf("positive rollback was rejected: %v", err)
	}
}

func TestDecodeDefaultsOmittedTelemetryWidthsToOne(t *testing.T) {
	encoded := mustActivationFrame(t)
	encoded = stripMetadataFields(t, encoded, "execution_batch_size", "verification_width")
	decoded, err := Unmarshal(encoded)
	if err != nil {
		t.Fatalf("unmarshal metadata defaults: %v", err)
	}
	if decoded.ExecutionBatchSize != 1 || decoded.VerificationWidth != 1 {
		t.Fatalf("telemetry defaults = %d/%d, want 1/1", decoded.ExecutionBatchSize, decoded.VerificationWidth)
	}
}

func TestMarshalRejectsInvalidV2MetadataValues(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Frame)
	}{
		{name: "negative execution batch", mutate: func(frame *Frame) { frame.ExecutionBatchSize = -1 }},
		{name: "negative verification width", mutate: func(frame *Frame) { frame.VerificationWidth = -1 }},
		{name: "non-boolean token EOG", mutate: func(frame *Frame) { frame.TokenEOG = []uint32{2} }},
		{name: "message byte overflow", mutate: func(frame *Frame) { frame.MessageBytes = []int{256} }},
		{name: "max tokens overflow", mutate: func(frame *Frame) { frame.MaxTokens = 1025 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			frame := Frame{Metadata: Metadata{
				SessionID: "s", RequestID: "r", ModelID: "m",
				StageIndex: 0, StageCount: 1, NodeName: "node",
				LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindText,
			}, Payload: []byte("hello")}
			test.mutate(&frame)
			if _, err := Marshal(frame); err == nil {
				t.Fatal("invalid StageWire v2 metadata was accepted")
			}
		})
	}
}

func TestMarshalRejectsPartialDeploymentIdentity(t *testing.T) {
	tests := []DeploymentIdentity{
		{DeploymentID: "deployment-a"},
		{DeploymentID: "deployment-a", Epoch: 1},
		{DeploymentID: "deployment-a", Epoch: 1, ModelSHA256: "not-a-digest"},
	}
	for _, identity := range tests {
		frame := Frame{
			Metadata: Metadata{
				SessionID: "s", RequestID: "r", ModelID: "m",
				DeploymentIdentity: identity,
				StageIndex:         0, StageCount: 1, NodeName: "node",
				LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindText,
			},
			Payload: []byte("hello"),
		}
		if _, err := Marshal(frame); err == nil {
			t.Fatalf("expected deployment identity rejection for %+v", identity)
		}
	}
}

func TestDecodeRejectsCorruptionAndTruncation(t *testing.T) {
	encoded := mustActivationFrame(t)

	badMagic := append([]byte(nil), encoded...)
	badMagic[0] = 'X'
	if _, err := Unmarshal(badMagic); !errors.Is(err, ErrInvalidMagic) {
		t.Fatalf("bad magic error=%v", err)
	}

	badVersion := append([]byte(nil), encoded...)
	binary.BigEndian.PutUint16(badVersion[4:6], 99)
	if _, err := Unmarshal(badVersion); !errors.Is(err, ErrUnsupportedVersion) {
		t.Fatalf("bad version error=%v", err)
	}

	truncated := encoded[:len(encoded)-1]
	if _, err := Unmarshal(truncated); !errors.Is(err, ErrTruncatedFrame) {
		t.Fatalf("truncated error=%v", err)
	}

	corrupted := append([]byte(nil), encoded...)
	corrupted[len(corrupted)-1] ^= 0xff
	if _, err := Unmarshal(corrupted); !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("checksum error=%v", err)
	}

	trailing := append(append([]byte(nil), encoded...), 1)
	if _, err := Unmarshal(trailing); !errors.Is(err, ErrTrailingData) {
		t.Fatalf("trailing error=%v", err)
	}
}

func TestMarshalRejectsTensorLengthMismatch(t *testing.T) {
	_, err := Marshal(Frame{
		Metadata: Metadata{
			SessionID: "s", RequestID: "r", ModelID: "m",
			StageIndex: 0, StageCount: 2, NodeName: "node",
			LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindActivation,
			DType: "f32", Shape: []int64{2, 2},
		},
		Payload: make([]byte, 12),
	})
	if err == nil {
		t.Fatal("expected tensor length mismatch")
	}
}

func mustActivationFrame(t *testing.T) []byte {
	t.Helper()
	encoded, err := Marshal(Frame{
		Metadata: Metadata{
			SessionID: "s", RequestID: "r", ModelID: "m",
			StageIndex: 0, StageCount: 2, NodeName: "node",
			LayerStart: 0, LayerEnd: 1, PayloadKind: PayloadKindActivation,
			DType: "f32", Shape: []int64{1},
		},
		Payload: []byte{0, 0, 0, 0},
	})
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return encoded
}

func stripMetadataFields(t *testing.T, encoded []byte, fields ...string) []byte {
	t.Helper()
	metadataLen := int(binary.BigEndian.Uint32(encoded[8:12]))
	var metadata map[string]any
	if err := json.Unmarshal(encoded[HeaderSize:HeaderSize+metadataLen], &metadata); err != nil {
		t.Fatalf("decode fixture metadata: %v", err)
	}
	for _, field := range fields {
		delete(metadata, field)
	}
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		t.Fatalf("encode fixture metadata: %v", err)
	}
	payload := encoded[HeaderSize+metadataLen:]
	header := append([]byte(nil), encoded[:HeaderSize]...)
	binary.BigEndian.PutUint32(header[8:12], uint32(len(metadataBytes)))
	return append(append(header, metadataBytes...), payload...)
}

func setMetadataField(t *testing.T, encoded []byte, field string, value any) []byte {
	t.Helper()
	metadataLen := int(binary.BigEndian.Uint32(encoded[8:12]))
	var metadata map[string]any
	if err := json.Unmarshal(encoded[HeaderSize:HeaderSize+metadataLen], &metadata); err != nil {
		t.Fatalf("decode fixture metadata: %v", err)
	}
	metadata[field] = value
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		t.Fatalf("encode fixture metadata: %v", err)
	}
	payload := encoded[HeaderSize+metadataLen:]
	header := append([]byte(nil), encoded[:HeaderSize]...)
	binary.BigEndian.PutUint32(header[8:12], uint32(len(metadataBytes)))
	return append(append(header, metadataBytes...), payload...)
}

func equalUint32s(left, right []uint32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
