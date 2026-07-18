package stageexec

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/inference"
)

func normalizeOrNewID(value string, prefix string) string {
	value = strings.TrimSpace(value)
	if value != "" {
		return value
	}
	return newID(prefix)
}

func newID(prefix string) string {
	var random [12]byte
	if _, err := rand.Read(random[:]); err == nil {
		return prefix + "-" + hex.EncodeToString(random[:])
	}
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func generationPassRequestID(rootRequestID string, phase inference.Phase, decodeStep int) string {
	return fmt.Sprintf("%s-%s-%d", rootRequestID, phase, decodeStep)
}

func stageOperationRequestID(passRequestID string, stageIndex int) string {
	return fmt.Sprintf("%s-stage-%d", passRequestID, stageIndex)
}
