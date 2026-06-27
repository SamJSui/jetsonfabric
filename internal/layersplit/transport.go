package layersplit

import (
	"context"
	"fmt"
	"math"
	"time"
)

type TransportKind string

const (
	TransportHTTP      TransportKind = "http"
	TransportGRPC      TransportKind = "grpc"
	TransportCustomTCP TransportKind = "custom_tcp"
	TransportLocal     TransportKind = "local"
)

type ActivationTransport interface {
	RunStage(ctx context.Context, target StageTarget, req ActivationRequest) (ActivationResponse, error)
}

type StageTarget struct {
	NodeID  string
	BaseURL string
	Stage   Stage
}

type ActivationRequest struct {
	SessionID   string    `json:"session_id"`
	RequestID   string    `json:"request_id"`
	ModelID     string    `json:"model_id"`
	StageIndex  int       `json:"stage_index"`
	NodeID      string    `json:"node_id"`
	Role        StageRole `json:"role"`
	LayerStart  int       `json:"layer_start"`
	LayerEnd    int       `json:"layer_end"`
	DecodeStep  int       `json:"decode_step"`
	Shape       []int     `json:"shape,omitempty"`
	DType       string    `json:"dtype,omitempty"`
	Payload     string    `json:"payload"`
	BytesIn     int       `json:"bytes_in"`
	Transport   string    `json:"transport"`
	RequestedAt time.Time `json:"requested_at,omitempty"`
}

type ActivationResponse struct {
	SessionID  string     `json:"session_id"`
	RequestID  string     `json:"request_id"`
	ModelID    string     `json:"model_id"`
	StageIndex int        `json:"stage_index"`
	NodeID     string     `json:"node_id"`
	Role       StageRole  `json:"role"`
	LayerStart int        `json:"layer_start"`
	LayerEnd   int        `json:"layer_end"`
	DecodeStep int        `json:"decode_step"`
	Shape      []int      `json:"shape,omitempty"`
	DType      string     `json:"dtype,omitempty"`
	Payload    string     `json:"payload"`
	BytesIn    int        `json:"bytes_in"`
	BytesOut   int        `json:"bytes_out"`
	Transport  string     `json:"transport"`
	LatencyMS  int64      `json:"latency_ms"`
	Trace      StageTrace `json:"trace"`
}

type StageTrace struct {
	StageIndex int       `json:"stage_index"`
	NodeID     string    `json:"node_id"`
	Role       StageRole `json:"role"`
	LayerStart int       `json:"layer_start"`
	LayerEnd   int       `json:"layer_end"`
	BytesIn    int       `json:"bytes_in"`
	BytesOut   int       `json:"bytes_out"`
	Transport  string    `json:"transport"`
	LatencyMS  int64     `json:"latency_ms"`
}

func NewTransport(kind TransportKind) (ActivationTransport, error) {
	switch kind {
	case "", TransportHTTP:
		return NewHTTPTransport(0), nil
	case TransportLocal:
		return &LocalTransport{}, nil
	case TransportGRPC:
		return nil, fmt.Errorf("layer split transport %q is not implemented yet", kind)
	case TransportCustomTCP:
		return nil, fmt.Errorf("layer split transport %q is not implemented yet", kind)
	default:
		return nil, fmt.Errorf("unknown layer split transport %q", kind)
	}
}

func BuildStageRequest(sessionID string, requestID string, modelID string, stage Stage, payload string, transport TransportKind) ActivationRequest {
	return ActivationRequest{
		SessionID:   sessionID,
		RequestID:   requestID,
		ModelID:     modelID,
		StageIndex:  stage.Index,
		NodeID:      stage.NodeID,
		Role:        stage.Role,
		LayerStart:  stage.LayerStart,
		LayerEnd:    stage.LayerEnd,
		DecodeStep:  0,
		Shape:       SyntheticActivationShape(stage),
		DType:       "synthetic",
		Payload:     payload,
		BytesIn:     len([]byte(payload)),
		Transport:   string(transport),
		RequestedAt: time.Now().UTC(),
	}
}

func SyntheticActivationShape(stage Stage) []int {
	layerCount := stage.LayerEnd - stage.LayerStart
	if layerCount < 1 {
		layerCount = 1
	}
	return []int{1, layerCount, 1}
}

func NormalizeResponse(req ActivationRequest, resp ActivationResponse, elapsed time.Duration) ActivationResponse {
	if resp.SessionID == "" {
		resp.SessionID = req.SessionID
	}
	if resp.RequestID == "" {
		resp.RequestID = req.RequestID
	}
	if resp.ModelID == "" {
		resp.ModelID = req.ModelID
	}
	if resp.StageIndex == 0 && req.StageIndex != 0 {
		resp.StageIndex = req.StageIndex
	}
	if resp.NodeID == "" {
		resp.NodeID = req.NodeID
	}
	if resp.Role == "" {
		resp.Role = req.Role
	}
	if resp.LayerStart == 0 && req.LayerStart != 0 {
		resp.LayerStart = req.LayerStart
	}
	if resp.LayerEnd == 0 {
		resp.LayerEnd = req.LayerEnd
	}
	if resp.Shape == nil {
		resp.Shape = req.Shape
	}
	if resp.DType == "" {
		resp.DType = req.DType
	}
	if resp.BytesIn == 0 {
		resp.BytesIn = req.BytesIn
	}
	if resp.BytesOut == 0 {
		resp.BytesOut = len([]byte(resp.Payload))
	}
	if resp.Transport == "" {
		resp.Transport = req.Transport
	}
	if resp.LatencyMS == 0 && elapsed > 0 {
		resp.LatencyMS = elapsed.Milliseconds()
	}
	resp.Trace = StageTrace{
		StageIndex: resp.StageIndex,
		NodeID:     resp.NodeID,
		Role:       resp.Role,
		LayerStart: resp.LayerStart,
		LayerEnd:   resp.LayerEnd,
		BytesIn:    resp.BytesIn,
		BytesOut:   resp.BytesOut,
		Transport:  resp.Transport,
		LatencyMS:  resp.LatencyMS,
	}
	return resp
}

func SyntheticStageResponse(req ActivationRequest, nodeID string, elapsed time.Duration) ActivationResponse {
	if nodeID == "" {
		nodeID = req.NodeID
	}
	payload := fmt.Sprintf("%s -> %s[%d:%d]", req.Payload, nodeID, req.LayerStart, req.LayerEnd)
	return NormalizeResponse(req, ActivationResponse{
		NodeID:     nodeID,
		Payload:    payload,
		BytesOut:   len([]byte(payload)),
		LatencyMS:  elapsed.Milliseconds(),
		Transport:  req.Transport,
		DecodeStep: req.DecodeStep,
	}, elapsed)
}

func ValidWeight(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
