package stagewire

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/inference"
)

const (
	Version          uint16 = 1
	HeaderSize              = 20
	MaxMetadataBytes        = 1 << 20
	MaxPayloadBytes  int64  = 512 << 20
)

var magic = [4]byte{'J', 'F', 'S', 'T'}

var (
	ErrInvalidMagic       = errors.New("invalid stagewire magic")
	ErrUnsupportedVersion = errors.New("unsupported stagewire version")
	ErrFrameTooLarge      = errors.New("stagewire frame is too large")
	ErrTruncatedFrame     = errors.New("truncated stagewire frame")
	ErrTrailingData       = errors.New("stagewire frame has trailing data")
	ErrChecksumMismatch   = errors.New("stagewire payload checksum mismatch")
)

func Marshal(frame Frame) ([]byte, error) {
	var out bytes.Buffer
	if err := Encode(&out, frame); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func Unmarshal(data []byte) (Frame, error) {
	return Decode(bytes.NewReader(data))
}

func Encode(w io.Writer, frame Frame) error {
	frame = normalizeFrame(frame)
	if err := Validate(frame); err != nil {
		return err
	}

	metadata, err := json.Marshal(frame.Metadata)
	if err != nil {
		return fmt.Errorf("encode stagewire metadata: %w", err)
	}
	if len(metadata) > MaxMetadataBytes || int64(len(frame.Payload)) > MaxPayloadBytes {
		return ErrFrameTooLarge
	}

	header := make([]byte, HeaderSize)
	copy(header[:4], magic[:])
	binary.BigEndian.PutUint16(header[4:6], Version)
	binary.BigEndian.PutUint16(header[6:8], 0)
	binary.BigEndian.PutUint32(header[8:12], uint32(len(metadata)))
	binary.BigEndian.PutUint64(header[12:20], uint64(len(frame.Payload)))

	if err := writeAll(w, header); err != nil {
		return err
	}
	if err := writeAll(w, metadata); err != nil {
		return err
	}
	if err := writeAll(w, frame.Payload); err != nil {
		return err
	}
	return nil
}

func Decode(r io.Reader) (Frame, error) {
	header := make([]byte, HeaderSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return Frame{}, fmt.Errorf("%w: header: %v", ErrTruncatedFrame, err)
	}
	if !bytes.Equal(header[:4], magic[:]) {
		return Frame{}, ErrInvalidMagic
	}
	version := binary.BigEndian.Uint16(header[4:6])
	if version != Version {
		return Frame{}, fmt.Errorf("%w: %d", ErrUnsupportedVersion, version)
	}
	if flags := binary.BigEndian.Uint16(header[6:8]); flags != 0 {
		return Frame{}, fmt.Errorf("stagewire flags must be zero, got %d", flags)
	}
	metadataLen := binary.BigEndian.Uint32(header[8:12])
	payloadLen := binary.BigEndian.Uint64(header[12:20])
	if metadataLen == 0 || metadataLen > MaxMetadataBytes || payloadLen > uint64(MaxPayloadBytes) {
		return Frame{}, ErrFrameTooLarge
	}
	if payloadLen > uint64(math.MaxInt) {
		return Frame{}, ErrFrameTooLarge
	}

	metadataBytes := make([]byte, int(metadataLen))
	if _, err := io.ReadFull(r, metadataBytes); err != nil {
		return Frame{}, fmt.Errorf("%w: metadata: %v", ErrTruncatedFrame, err)
	}
	payload := make([]byte, int(payloadLen))
	if _, err := io.ReadFull(r, payload); err != nil {
		return Frame{}, fmt.Errorf("%w: payload: %v", ErrTruncatedFrame, err)
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); n != 0 || (err != nil && err != io.EOF) {
		return Frame{}, ErrTrailingData
	}

	decoder := json.NewDecoder(bytes.NewReader(metadataBytes))
	decoder.DisallowUnknownFields()
	var metadata Metadata
	if err := decoder.Decode(&metadata); err != nil {
		return Frame{}, fmt.Errorf("decode stagewire metadata: %w", err)
	}
	frame := Frame{Metadata: metadata, Payload: payload}
	if metadata.ProtocolVersion != version {
		return Frame{}, fmt.Errorf("metadata protocol_version %d does not match frame version %d", metadata.ProtocolVersion, version)
	}
	if err := Validate(frame); err != nil {
		return Frame{}, err
	}
	return frame, nil
}

func Validate(frame Frame) error {
	m := frame.Metadata
	if m.ProtocolVersion != Version {
		return fmt.Errorf("protocol_version must be %d", Version)
	}
	if !m.Operation.Valid() {
		return fmt.Errorf("invalid operation %q", m.Operation)
	}
	if strings.TrimSpace(m.SessionID) == "" {
		return errors.New("session_id is required")
	}
	if strings.TrimSpace(m.RequestID) == "" {
		return errors.New("request_id is required")
	}
	if strings.TrimSpace(m.ModelID) == "" {
		return errors.New("model_id is required")
	}
	if err := validateDeploymentIdentity(m.DeploymentIdentity); err != nil {
		return err
	}
	if !m.Phase.Valid() {
		return fmt.Errorf("invalid phase %q", m.Phase)
	}
	if err := m.Position().Validate(); err != nil {
		return err
	}
	if m.LayerStart < 0 || m.LayerEnd <= m.LayerStart {
		return fmt.Errorf("invalid layer range [%d:%d)", m.LayerStart, m.LayerEnd)
	}
	if !m.PayloadKind.Valid() {
		return fmt.Errorf("invalid payload_kind %q", m.PayloadKind)
	}
	if m.PayloadBytes != int64(len(frame.Payload)) {
		return fmt.Errorf("payload_bytes=%d does not match payload length %d", m.PayloadBytes, len(frame.Payload))
	}
	if m.PayloadBytes > MaxPayloadBytes {
		return ErrFrameTooLarge
	}
	if got := crc32.ChecksumIEEE(frame.Payload); got != m.PayloadCRC32 {
		return fmt.Errorf("%w: metadata=%08x actual=%08x", ErrChecksumMismatch, m.PayloadCRC32, got)
	}
	if m.Transport != Transport {
		return fmt.Errorf("transport must be %q", Transport)
	}
	return validatePayloadMetadata(m, frame.Payload)
}

func validateDeploymentIdentity(identity DeploymentIdentity) error {
	if !identity.Present() {
		return nil
	}
	if strings.TrimSpace(identity.DeploymentID) == "" || identity.Epoch == 0 {
		return errors.New("managed stagewire identity requires deployment_id and positive deployment_epoch")
	}
	if len(identity.ModelSHA256) != 64 {
		return errors.New("managed stagewire identity requires a 64-character model_sha256")
	}
	if _, err := hex.DecodeString(identity.ModelSHA256); err != nil {
		return errors.New("managed stagewire identity requires a hexadecimal model_sha256")
	}
	return nil
}

func normalizeFrame(frame Frame) Frame {
	frame.ProtocolVersion = Version
	if frame.Operation == "" {
		frame.Operation = OperationExecute
	}
	if frame.Phase == "" {
		if frame.DecodeStep == 0 {
			frame.Phase = inference.PhasePrefill
		} else {
			frame.Phase = inference.PhaseDecode
		}
	}
	if frame.PayloadKind == PayloadKindText && frame.Encoding == "" {
		frame.Encoding = "utf-8"
	}
	if isTensorKind(frame.PayloadKind) {
		if frame.ByteOrder == "" {
			frame.ByteOrder = "little"
		}
		if frame.Layout == "" {
			frame.Layout = "row_major"
		}
	}
	frame.PayloadBytes = int64(len(frame.Payload))
	frame.PayloadCRC32 = crc32.ChecksumIEEE(frame.Payload)
	frame.Transport = Transport
	return frame
}

func validatePayloadMetadata(m Metadata, payload []byte) error {
	if m.Error != "" {
		return nil
	}
	switch m.PayloadKind {
	case PayloadKindText:
		if m.Encoding != "utf-8" {
			return fmt.Errorf("text payload encoding must be utf-8")
		}
		if len(m.Shape) != 0 || m.DType != "" || m.ByteOrder != "" || m.Layout != "" {
			return errors.New("text payload must not declare tensor metadata")
		}
	case PayloadKindTokens, PayloadKindActivation, PayloadKindSampledToken:
		if strings.TrimSpace(m.DType) == "" {
			return errors.New("tensor payload dtype is required")
		}
		if len(m.Shape) == 0 {
			return errors.New("tensor payload shape is required")
		}
		if m.ByteOrder != "little" {
			return errors.New("tensor payload byte_order must be little")
		}
		if m.Layout != "row_major" {
			return errors.New("tensor payload layout must be row_major")
		}
		expected, err := tensorByteLength(m.DType, m.Shape)
		if err != nil {
			return err
		}
		if expected != int64(len(payload)) {
			return fmt.Errorf("tensor metadata requires %d bytes, got %d", expected, len(payload))
		}
	}
	return nil
}

func tensorByteLength(dtype string, shape []int64) (int64, error) {
	width, ok := map[string]int64{
		"u8":   1,
		"i8":   1,
		"f16":  2,
		"bf16": 2,
		"i32":  4,
		"u32":  4,
		"f32":  4,
		"i64":  8,
		"u64":  8,
		"f64":  8,
	}[dtype]
	if !ok {
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
	count := int64(1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("shape dimensions must be positive, got %d", dim)
		}
		if count > MaxPayloadBytes/dim {
			return 0, ErrFrameTooLarge
		}
		count *= dim
	}
	if count > MaxPayloadBytes/width {
		return 0, ErrFrameTooLarge
	}
	return count * width, nil
}

func isTensorKind(kind PayloadKind) bool {
	return kind == PayloadKindTokens || kind == PayloadKindActivation || kind == PayloadKindSampledToken
}

func writeAll(w io.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}
