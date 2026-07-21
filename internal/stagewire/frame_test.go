package stagewire

import (
	"bytes"
	"encoding/binary"
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
