// Package stagewire defines the versioned binary contract used for one
// JetsonFabric stage operation. It owns framing and transport metadata, while
// internal/inference owns semantic lifecycle and payload-transition rules.
package stagewire

import "github.com/SamJSui/jetsonfabric/internal/inference"

const (
	ContentType = "application/vnd.jetsonfabric.stage.v1+octet-stream"
	Transport   = "http_binary_v1"
)

type Operation string

const (
	OperationExecute      Operation = "execute"
	OperationCloseSession Operation = "close_session"
)

// Empty remains a valid legacy value and is interpreted as execute. Encoders
// normalize new frames to the explicit execute operation.
func (o Operation) Valid() bool {
	return o == "" || o == OperationExecute || o == OperationCloseSession
}

type PayloadKind = inference.PayloadKind

const (
	PayloadKindText         = inference.PayloadKindText
	PayloadKindTokens       = inference.PayloadKindTokens
	PayloadKindActivation   = inference.PayloadKindActivation
	PayloadKindSampledToken = inference.PayloadKindSampledToken
)

type DeploymentIdentity struct {
	DeploymentID string `json:"deployment_id,omitempty"`
	Epoch        uint64 `json:"deployment_epoch,omitempty"`
	ModelSHA256  string `json:"model_sha256,omitempty"`
}

func (d DeploymentIdentity) Present() bool {
	return d.DeploymentID != "" || d.Epoch != 0 || d.ModelSHA256 != ""
}

// Metadata is encoded as JSON inside a stagewire frame. Payload bytes follow the
// metadata directly and are never base64-encoded or represented as JSON arrays.
type Metadata struct {
	ProtocolVersion uint16 `json:"protocol_version"`

	Operation Operation `json:"operation,omitempty"`
	SessionID string    `json:"session_id"`
	RequestID string    `json:"request_id"`
	ModelID   string    `json:"model_id"`
	DeploymentIdentity

	Phase      inference.Phase `json:"phase"`
	DecodeStep int             `json:"decode_step"`

	StageIndex int    `json:"stage_index"`
	StageCount int    `json:"stage_count"`
	NodeName   string `json:"node_name"`

	LayerStart int `json:"layer_start"`
	LayerEnd   int `json:"layer_end"`

	PayloadKind PayloadKind `json:"payload_kind"`
	Encoding    string      `json:"encoding,omitempty"`
	DType       string      `json:"dtype,omitempty"`
	Shape       []int64     `json:"shape,omitempty"`
	ByteOrder   string      `json:"byte_order,omitempty"`
	Layout      string      `json:"layout,omitempty"`

	PayloadBytes int64  `json:"payload_bytes"`
	PayloadCRC32 uint32 `json:"payload_crc32"`
	Transport    string `json:"transport"`

	MaxTokens int `json:"max_tokens,omitempty"`

	BytesIn          int64  `json:"bytes_in,omitempty"`
	BytesOut         int64  `json:"bytes_out,omitempty"`
	PromptTokens     int    `json:"prompt_tokens,omitempty"`
	CompletionTokens int    `json:"completion_tokens,omitempty"`
	LatencyMS        int    `json:"latency_ms,omitempty"`
	Error            string `json:"error,omitempty"`
	Message          string `json:"message,omitempty"`
}

func (m Metadata) Position() inference.StagePosition {
	return inference.StagePosition{Index: m.StageIndex, Count: m.StageCount}
}

func (m Metadata) IsFirstStage() bool { return m.Position().IsFirst() }
func (m Metadata) IsLastStage() bool  { return m.Position().IsLast() }

// Frame is the in-process representation of one stagewire message. Metadata is
// embedded so existing call sites can continue to access fields directly.
type Frame struct {
	Metadata
	Payload []byte `json:"-"`
}

type StageRequest = Frame
type StageResponse = Frame
