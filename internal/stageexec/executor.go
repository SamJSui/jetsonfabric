package stageexec

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/inference"
	"github.com/SamJSui/jetsonfabric/internal/stagewire"
)

const defaultHTTPTimeout = 2 * time.Minute

type Config struct {
	Client       *http.Client
	ClusterToken string
}

type Executor struct {
	client       *http.Client
	clusterToken string
}

type Request struct {
	RequestID  string
	SessionID  string
	Model      string
	Payload    string
	Data       []byte
	Kind       stagewire.PayloadKind
	Phase      inference.Phase
	DecodeStep int
	MaxTokens  int
	Deployment stagewire.DeploymentIdentity
	Plan       clusterplan.RoutePreview
}

type Result struct {
	RequestID        string                `json:"request_id"`
	SessionID        string                `json:"session_id"`
	Model            string                `json:"model"`
	PayloadKind      stagewire.PayloadKind `json:"payload_kind"`
	Payload          string                `json:"payload,omitempty"`
	PayloadBytes     int                   `json:"payload_bytes"`
	SampledToken     *uint32               `json:"sampled_token,omitempty"`
	TokenText        string                `json:"token_text,omitempty"`
	EndOfGeneration  bool                  `json:"end_of_generation,omitempty"`
	GeneratedText    string                `json:"generated_text,omitempty"`
	SampledTokens    []uint32              `json:"sampled_tokens,omitempty"`
	FinishReason     string                `json:"finish_reason,omitempty"`
	PromptTokens     int                   `json:"prompt_tokens,omitempty"`
	CompletionTokens int                   `json:"completion_tokens,omitempty"`
	BytesIn          int64                 `json:"bytes_in,omitempty"`
	BytesOut         int64                 `json:"bytes_out,omitempty"`
	Stages           []StageTrace          `json:"stages"`

	Data []byte `json:"-"`
}

type StageTrace struct {
	RequestID       string                `json:"request_id"`
	SessionID       string                `json:"session_id"`
	StageIndex      int                   `json:"stage_index"`
	StageCount      int                   `json:"stage_count"`
	NodeID          string                `json:"node_id"`
	NodeName        string                `json:"node_name"`
	APIURL          string                `json:"api_url"`
	StatusCode      int                   `json:"status_code"`
	LatencyMS       int                   `json:"latency_ms"`
	BytesIn         int64                 `json:"bytes_in"`
	BytesOut        int64                 `json:"bytes_out"`
	PayloadKindIn   stagewire.PayloadKind `json:"payload_kind_in"`
	PayloadKindOut  stagewire.PayloadKind `json:"payload_kind_out"`
	PayloadIn       int                   `json:"payload_in"`
	PayloadOut      int                   `json:"payload_out"`
	PayloadCRC32In  uint32                `json:"payload_crc32_in"`
	PayloadCRC32Out uint32                `json:"payload_crc32_out"`
	Transport       string                `json:"transport"`
	Operation       stagewire.Operation   `json:"operation"`
	Phase           inference.Phase       `json:"phase"`
	DecodeStep      int                   `json:"decode_step"`
}

type StageRequest = stagewire.StageRequest
type StageResponse = stagewire.StageResponse

type StageError struct {
	StageIndex int
	StatusCode int
	Code       string
	Message    string
}

func (e StageError) Error() string {
	if e.Code == "" && e.Message == "" {
		return fmt.Sprintf("stage %d failed with HTTP %d", e.StageIndex, e.StatusCode)
	}
	if e.Message == "" {
		return fmt.Sprintf("stage %d failed with HTTP %d: %s", e.StageIndex, e.StatusCode, e.Code)
	}
	return fmt.Sprintf("stage %d failed with HTTP %d: %s: %s", e.StageIndex, e.StatusCode, e.Code, e.Message)
}

func New(cfg Config) *Executor {
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: defaultHTTPTimeout}
	}
	return &Executor{client: client, clusterToken: strings.TrimSpace(cfg.ClusterToken)}
}

// Execute runs exactly one ordered prefill or decode pass. Payload transition
// validation is always enabled for the pipeline path.
func (e *Executor) Execute(ctx context.Context, req Request) (Result, error) {
	if e == nil {
		e = New(Config{})
	}
	if len(req.Plan.Stages) == 0 {
		return Result{}, fmt.Errorf("stage plan is empty")
	}
	requestID := normalizeOrNewID(req.RequestID, "request")
	sessionID := normalizeOrNewID(req.SessionID, "session")
	phase := req.Phase
	if phase == "" {
		phase = inference.PhasePrefill
	}
	kind := req.Kind
	if kind == "" {
		kind = stagewire.PayloadKindText
	}
	payload := append([]byte(nil), req.Data...)
	if payload == nil {
		payload = []byte(req.Payload)
	}
	payloadMetadata := stagewire.Metadata{PayloadKind: kind}
	switch kind {
	case stagewire.PayloadKindText:
		payloadMetadata.Encoding = "utf-8"
	case stagewire.PayloadKindSampledToken:
		payloadMetadata.DType = "u32"
		payloadMetadata.Shape = []int64{1}
		payloadMetadata.ByteOrder = "little"
		payloadMetadata.Layout = "row_major"
	case stagewire.PayloadKindTokens:
		if len(payload)%4 != 0 || len(payload) == 0 {
			return Result{}, fmt.Errorf("token payload must contain one or more 32-bit token IDs")
		}
		payloadMetadata.DType = "i32"
		payloadMetadata.Shape = []int64{int64(len(payload) / 4)}
		payloadMetadata.ByteOrder = "little"
		payloadMetadata.Layout = "row_major"
	}

	result := Result{
		RequestID: requestID,
		SessionID: sessionID,
		Model:     req.Model,
		Stages:    make([]StageTrace, 0, len(req.Plan.Stages)),
	}
	var finalResponse StageResponse

	for _, stage := range req.Plan.Stages {
		stageRequestID := stageOperationRequestID(requestID, stage.StageIndex)
		stageReq := buildStageRequest(sessionID, stageRequestID, req.Model, req.Deployment, phase, req.DecodeStep, payloadMetadata, payload, req.MaxTokens, stage)
		stageResp, status, trace, err := e.callStage(ctx, stage, stageReq)
		trace.StageIndex = stage.StageIndex
		trace.StageCount = stage.StageCount
		trace.NodeID = stage.NodeID
		trace.NodeName = stage.NodeName
		trace.APIURL = stage.APIURL
		result.Stages = append(result.Stages, trace)
		if err != nil {
			return result, err
		}
		if status < 200 || status >= 300 {
			return result, StageError{StageIndex: stage.StageIndex, StatusCode: status, Code: stageResp.Error, Message: stageResp.Message}
		}
		if err := validateStageResponseIdentity(stageReq, stageResp); err != nil {
			return result, fmt.Errorf("stage %d response identity: %w", stage.StageIndex, err)
		}
		if err := inference.ValidatePayloadTransition(phase, stageReq.Position(), stageReq.PayloadKind, stageResp.PayloadKind); err != nil {
			return result, fmt.Errorf("stage %d payload contract: %w", stage.StageIndex, err)
		}
		payload = append(payload[:0], stageResp.Payload...)
		kind = stageResp.PayloadKind
		payloadMetadata = stagewire.Metadata{
			PayloadKind: stageResp.PayloadKind,
			Encoding:    stageResp.Encoding,
			DType:       stageResp.DType,
			Shape:       append([]int64(nil), stageResp.Shape...),
			ByteOrder:   stageResp.ByteOrder,
			Layout:      stageResp.Layout,
		}
		result.PromptTokens += stageResp.PromptTokens
		result.CompletionTokens += stageResp.CompletionTokens
		result.BytesIn += stageResp.BytesIn
		result.BytesOut += stageResp.BytesOut
		finalResponse = stageResp
	}

	result.PayloadKind = kind
	result.PayloadBytes = len(payload)
	result.Data = append([]byte(nil), payload...)
	result.TokenText = finalResponse.Message
	result.EndOfGeneration = kind == stagewire.PayloadKindSampledToken && finalResponse.CompletionTokens == 0
	if kind == stagewire.PayloadKindText {
		result.Payload = string(payload)
	}
	if kind == stagewire.PayloadKindSampledToken && len(payload) == 4 {
		token := binary.LittleEndian.Uint32(payload)
		result.SampledToken = &token
	}
	return result, nil
}

func buildStageRequest(sessionID string, requestID string, model string, deployment stagewire.DeploymentIdentity, phase inference.Phase, decodeStep int, payloadMetadata stagewire.Metadata, payload []byte, maxTokens int, stage clusterplan.Stage) StageRequest {
	metadata := stagewire.Metadata{
		Operation:          stagewire.OperationExecute,
		SessionID:          sessionID,
		RequestID:          requestID,
		ModelID:            model,
		DeploymentIdentity: deployment,
		Phase:              phase,
		DecodeStep:         decodeStep,
		StageIndex:         stage.StageIndex,
		StageCount:         stage.StageCount,
		NodeName:           stage.NodeName,
		LayerStart:         stage.LayerStart,
		LayerEnd:           stage.LayerEnd,
		PayloadKind:        payloadMetadata.PayloadKind,
		Encoding:           payloadMetadata.Encoding,
		DType:              payloadMetadata.DType,
		Shape:              append([]int64(nil), payloadMetadata.Shape...),
		ByteOrder:          payloadMetadata.ByteOrder,
		Layout:             payloadMetadata.Layout,
		MaxTokens:          maxTokens,
	}
	return StageRequest{Metadata: metadata, Payload: append([]byte(nil), payload...)}
}

func buildCloseSessionRequest(sessionID string, model string, deployment stagewire.DeploymentIdentity, stage clusterplan.Stage) StageRequest {
	return StageRequest{Metadata: stagewire.Metadata{
		Operation:          stagewire.OperationCloseSession,
		SessionID:          sessionID,
		RequestID:          stageOperationRequestID(sessionID+"-close", stage.StageIndex),
		ModelID:            model,
		DeploymentIdentity: deployment,
		Phase:              inference.PhasePrefill,
		StageIndex:         stage.StageIndex,
		StageCount:         stage.StageCount,
		NodeName:           stage.NodeName,
		LayerStart:         stage.LayerStart,
		LayerEnd:           stage.LayerEnd,
		PayloadKind:        stagewire.PayloadKindText,
		Encoding:           "utf-8",
		MaxTokens:          1,
	}}
}

func validateStageResponseIdentity(request StageRequest, response StageResponse) error {
	if response.SessionID != request.SessionID ||
		response.RequestID != request.RequestID ||
		response.ModelID != request.ModelID ||
		response.DeploymentIdentity != request.DeploymentIdentity ||
		response.Phase != request.Phase ||
		response.DecodeStep != request.DecodeStep ||
		response.StageIndex != request.StageIndex ||
		response.StageCount != request.StageCount ||
		response.NodeName != request.NodeName ||
		response.LayerStart != request.LayerStart ||
		response.LayerEnd != request.LayerEnd {
		return fmt.Errorf("does not match request")
	}
	return nil
}

func (e *Executor) callStage(ctx context.Context, stage clusterplan.Stage, stageReq StageRequest) (StageResponse, int, StageTrace, error) {
	body, err := stagewire.Marshal(stageReq)
	if err != nil {
		return StageResponse{}, 0, StageTrace{}, err
	}
	endpoint, err := stageEndpoint(stage.APIURL)
	if err != nil {
		return StageResponse{}, 0, StageTrace{}, err
	}
	outbound, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return StageResponse{}, 0, StageTrace{}, err
	}
	outbound.Header.Set("Content-Type", stagewire.ContentType)
	outbound.Header.Set("Accept", stagewire.ContentType)
	outbound.Header.Set(api.HeaderClusterToken, e.clusterToken)
	outbound.ContentLength = int64(len(body))

	baseTrace := StageTrace{
		RequestID:      stageReq.RequestID,
		SessionID:      stageReq.SessionID,
		PayloadKindIn:  stageReq.PayloadKind,
		PayloadIn:      len(stageReq.Payload),
		PayloadCRC32In: crc32.ChecksumIEEE(stageReq.Payload),
		Transport:      stagewire.Transport,
		Operation:      stageReq.Operation,
		Phase:          stageReq.Phase,
		DecodeStep:     stageReq.DecodeStep,
	}
	resp, err := e.client.Do(outbound)
	if err != nil {
		return StageResponse{}, 0, baseTrace, err
	}
	defer resp.Body.Close()

	stageResp, err := decodeStageResponse(resp)
	if err != nil {
		baseTrace.StatusCode = resp.StatusCode
		return StageResponse{}, resp.StatusCode, baseTrace, err
	}
	trace := baseTrace
	trace.StatusCode = resp.StatusCode
	trace.LatencyMS = stageResp.LatencyMS
	trace.BytesIn = stageResp.BytesIn
	trace.BytesOut = stageResp.BytesOut
	trace.PayloadKindOut = stageResp.PayloadKind
	trace.PayloadOut = len(stageResp.Payload)
	trace.PayloadCRC32Out = stageResp.PayloadCRC32
	return stageResp, resp.StatusCode, trace, nil
}

func decodeStageResponse(resp *http.Response) (StageResponse, error) {
	if matchesStagewireMediaType(resp.Header.Get("Content-Type")) {
		return stagewire.Decode(resp.Body)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return StageResponse{}, err
	}
	var failure struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &failure); err != nil {
		return StageResponse{}, fmt.Errorf("decode stage response with content-type %q: %w: %s", resp.Header.Get("Content-Type"), err, string(body))
	}
	return StageResponse{Metadata: stagewire.Metadata{Error: failure.Error, Message: failure.Message}}, nil
}

func matchesStagewireMediaType(value string) bool {
	mediaType, _, err := mime.ParseMediaType(value)
	return err == nil && strings.EqualFold(mediaType, stagewire.ContentType)
}

func stageEndpoint(apiURL string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(apiURL))
	if err != nil {
		return "", fmt.Errorf("parse stage API URL: %w", err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("stage API URL must include scheme and host")
	}
	parsed.Path = joinPath(parsed.Path, api.PathLayerSplitStage)
	parsed.RawQuery = ""
	return parsed.String(), nil
}

func joinPath(base string, suffix string) string {
	base = strings.TrimRight(base, "/")
	suffix = "/" + strings.TrimLeft(suffix, "/")
	if base == "" {
		return suffix
	}
	return base + suffix
}
