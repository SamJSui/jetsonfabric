package stagewire

import (
	"testing"

	"github.com/SamJSui/jetsonfabric/internal/inference"
)

func TestFrameNormalizesExecuteOperation(t *testing.T) {
	frame := Frame{Metadata: Metadata{
		SessionID:   "session",
		RequestID:   "request",
		ModelID:     "model",
		Phase:       inference.PhasePrefill,
		StageIndex:  0,
		StageCount:  1,
		NodeName:    "node",
		LayerStart:  0,
		LayerEnd:    1,
		PayloadKind: PayloadKindText,
		Encoding:    "utf-8",
	}, Payload: []byte("prompt")}
	encoded, err := Marshal(frame)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := Unmarshal(encoded)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.Operation != OperationExecute {
		t.Fatalf("operation=%q", decoded.Operation)
	}
}

func TestCloseSessionOperationAllowsEmptyTextPayload(t *testing.T) {
	frame := Frame{Metadata: Metadata{
		Operation:   OperationCloseSession,
		SessionID:   "session",
		RequestID:   "request-close",
		ModelID:     "model",
		Phase:       inference.PhasePrefill,
		StageIndex:  0,
		StageCount:  2,
		NodeName:    "node",
		LayerStart:  0,
		LayerEnd:    1,
		PayloadKind: PayloadKindText,
		Encoding:    "utf-8",
	}}
	if _, err := Marshal(frame); err != nil {
		t.Fatalf("marshal close session: %v", err)
	}
}
