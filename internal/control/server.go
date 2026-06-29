package control

import (
	"cmp"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/SamJSui/jetsonfabric/internal/api"
	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
	"github.com/SamJSui/jetsonfabric/internal/modelregistry"
	"github.com/SamJSui/jetsonfabric/internal/routing"
	"github.com/SamJSui/jetsonfabric/internal/runtimeclient"
)

type Server struct {
	joinToken          string
	registry           modelregistry.Registry
	backendFactory     BackendFactory
	benchmarkRecorder  benchmarks.Recorder
	layerTransport     layersplit.ActivationTransport
	layerTransportKind layersplit.TransportKind
	now                func() time.Time
	mu                 sync.RWMutex
	nodes              map[string]cluster.NodeRecord
}

type BackendFactory func(cluster.RuntimeBackend) (runtimeclient.ChatBackend, error)

type Option func(*Server)

func WithBackendFactory(factory BackendFactory) Option {
	return func(s *Server) {
		s.backendFactory = factory
	}
}

func WithBenchmarkRecorder(recorder benchmarks.Recorder) Option {
	return func(s *Server) {
		s.benchmarkRecorder = recorder
	}
}

func WithLayerTransport(kind layersplit.TransportKind, transport layersplit.ActivationTransport) Option {
	return func(s *Server) {
		s.layerTransportKind = kind
		s.layerTransport = transport
	}
}

func WithClock(now func() time.Time) Option {
	return func(s *Server) {
		s.now = now
	}
}

func NewServer(joinToken string, registry modelregistry.Registry, opts ...Option) *Server {
	server := &Server{
		joinToken:          joinToken,
		registry:           registry,
		backendFactory:     defaultBackendFactory,
		benchmarkRecorder:  benchmarks.NoopRecorder{},
		layerTransportKind: layersplit.TransportHTTP,
		now:                func() time.Time { return time.Now().UTC() },
		nodes:              make(map[string]cluster.NodeRecord),
	}
	for _, opt := range opts {
		opt(server)
	}
	if server.backendFactory == nil {
		server.backendFactory = defaultBackendFactory
	}
	if server.benchmarkRecorder == nil {
		server.benchmarkRecorder = benchmarks.NoopRecorder{}
	}
	if server.layerTransport == nil {
		transport, err := layersplit.NewTransport(server.layerTransportKind)
		if err != nil {
			panic(err)
		}
		server.layerTransport = transport
	}
	if server.now == nil {
		server.now = func() time.Time { return time.Now().UTC() }
	}
	return server
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.RouteHealth, s.handleHealth)
	mux.HandleFunc(api.RouteNodes, s.handleNodes)
	mux.HandleFunc(api.RouteModels, s.handleModels)
	mux.HandleFunc(api.RoutePreview, s.handleRoutePreview)
	mux.HandleFunc(api.RouteLayerSplitPlan, s.handleLayerSplitPlan)
	mux.HandleFunc(api.RouteLayerSplitChat, s.handleLayerSplitCompletions)
	mux.HandleFunc(api.RouteAgentHeartbeat, s.handleHeartbeat)
	mux.HandleFunc(api.RouteChatCompletions, s.handleChatCompletions)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "jetsonfabric-control"})
}

func (s *Server) handleNodes(w http.ResponseWriter, _ *http.Request) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)
	writeJSON(w, http.StatusOK, map[string]any{"nodes": nodes})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.registry)
}

func (s *Server) handleRoutePreview(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("model")
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeJSON(w, http.StatusOK, routing.UnknownModel(modelID))
		return
	}
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	writeJSON(w, http.StatusOK, routing.Preview(model, nodes))
}

func (s *Server) handleLayerSplitPlan(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSpace(r.URL.Query().Get("model"))
	if modelID == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", modelID))
		return
	}
	if !slices.Contains(model.PlacementModes, cluster.RouteModeLayerSplit) {
		writeError(w, http.StatusBadRequest, errorLayerSplitUnsupported, fmt.Sprintf("model %q does not allow layer_split placement", model.ID))
		return
	}

	candidates := s.layerSplitCandidates(model)
	plan, err := layersplit.PlanForModel(model, candidates)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoLayerSplitRoute, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, plan)
}

func (s *Server) handleLayerSplitCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req chat.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "messages must contain at least one message")
		return
	}

	model, ok := s.registry.Find(req.Model)
	if !ok {
		writeError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", req.Model))
		return
	}
	if !slices.Contains(model.PlacementModes, cluster.RouteModeLayerSplit) {
		writeError(w, http.StatusBadRequest, errorLayerSplitUnsupported, fmt.Sprintf("model %q does not allow layer_split placement", model.ID))
		return
	}

	candidates := s.layerSplitCandidates(model)
	plan, err := layersplit.PlanForModel(model, candidates)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoLayerSplitRoute, err.Error())
		return
	}

	start := time.Now()
	sessionID := fmt.Sprintf("layer-session-%d", s.now().UnixNano())
	requestID := fmt.Sprintf("layer-request-%d", s.now().UnixNano())
	payload := lastMessageContent(req.Messages)
	if strings.TrimSpace(payload) == "" {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "last message content is required")
		return
	}

	stageResponses := make([]layersplit.ActivationResponse, 0, len(plan.Stages))
	for _, stage := range plan.Stages {
		stageReq := layersplit.BuildStageRequest(sessionID, requestID, model.ID, stage, payload, s.layerTransportKind)
		resp, err := s.layerTransport.RunStage(r.Context(), layersplit.StageTarget{
			NodeName: stage.NodeName,
			BaseURL:  stage.BaseURL,
			Stage:    stage,
		}, stageReq)
		if err != nil {
			writeError(w, http.StatusBadGateway, errorLayerSplitStageFailed, err.Error())
			return
		}
		stageResponses = append(stageResponses, resp)
		payload = resp.Payload
	}

	latency := time.Since(start)
	content := fmt.Sprintf("synthetic layer_split response: %s", payload)
	outputTokens := len(strings.Fields(content))
	resp := chat.CompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: s.now().Unix(),
		Model:   model.ID,
		Choices: []chat.Choice{
			{
				Index: 0,
				Message: chat.Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: &chat.Usage{
			PromptTokens:     len(strings.Fields(lastMessageContent(req.Messages))),
			CompletionTokens: outputTokens,
			TotalTokens:      len(strings.Fields(lastMessageContent(req.Messages))) + outputTokens,
		},
		Route: s.layerSplitRouteMetadata(plan, stageResponses, latency),
	}
	if err := s.benchmarkRecorder.Record(r.Context(), benchmarks.Record{
		Timestamp:    s.now(),
		ModelID:      model.ID,
		NodeName:     strings.Join(stageNodeNames(stageResponses), ","),
		RouteMode:    cluster.RouteModeLayerSplit,
		BackendID:    "layer-split",
		BackendKind:  cluster.RuntimeKindLlamaCPP,
		LatencyMS:    latency.Milliseconds(),
		OutputTokens: outputTokens,
	}); err != nil {
		writeError(w, http.StatusInternalServerError, errorBenchmarkRecordFailed, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeError(w, http.StatusUnauthorized, errorUnauthorized, "agent join token is missing or invalid")
		return
	}
	defer r.Body.Close()
	var heartbeat cluster.HeartbeatRequest
	if err := json.NewDecoder(r.Body).Decode(&heartbeat); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	if heartbeat.NodeName == "" {
		writeError(w, http.StatusBadRequest, errorMissingNodeName, "node_name is required")
		return
	}
	record := cluster.NodeRecord{
		NodeName:     heartbeat.NodeName,
		Hostname:     fallback(heartbeat.Hostname, heartbeat.NodeName),
		Arch:         fallback(heartbeat.Arch, "unknown"),
		OS:           fallback(heartbeat.OS, "unknown"),
		Capabilities: fallbackMap(heartbeat.Capabilities),
		Metrics:      fallbackMap(heartbeat.Metrics),
		Backends:     heartbeat.Backends,
		LastSeen:     s.now(),
	}
	s.mu.Lock()
	s.nodes[record.NodeName] = record
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"status": "registered", "node": record})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	var req chat.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, errorInvalidJSON, "request body must be valid JSON")
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, errorMissingModel, "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, errorMissingMessages, "messages must contain at least one message")
		return
	}
	model, ok := s.registry.Find(req.Model)
	if !ok {
		writeError(w, http.StatusBadRequest, errorUnknownModel, fmt.Sprintf("model %q is not in the registry", req.Model))
		return
	}
	node, backend, err := s.selectSingleNodeBackend(model)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoSingleNodeRoute, err.Error())
		return
	}

	chatBackend, err := s.backendFactory(backend)
	if err != nil {
		writeError(w, http.StatusBadGateway, errorBackendConfigInvalid, err.Error())
		return
	}
	resp, stats, err := chatBackend.Complete(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, errorBackendRequestFailed, err.Error())
		return
	}
	resp.Route = &chat.RouteMetadata{
		Mode:        cluster.RouteModeSingleNode,
		NodeName:    node.NodeName,
		BackendID:   backend.ID,
		BackendKind: backend.Kind,
		LatencyMS:   stats.Latency.Milliseconds(),
	}

	record := benchmarks.Record{
		Timestamp:    s.now(),
		ModelID:      model.ID,
		NodeName:     node.NodeName,
		RouteMode:    cluster.RouteModeSingleNode,
		BackendID:    backend.ID,
		BackendKind:  backend.Kind,
		LatencyMS:    stats.Latency.Milliseconds(),
		OutputTokens: stats.OutputTokens,
		TokensPerSec: stats.TokensPerSec,
		MemoryGB:     optionalFloat(node.Capabilities, cluster.CapabilityMemoryGB),
		TemperatureC: optionalFloat(node.Metrics, cluster.MetricTemperatureC),
	}
	if err := s.benchmarkRecorder.Record(r.Context(), record); err != nil {
		writeError(w, http.StatusInternalServerError, errorBenchmarkRecordFailed, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) authorized(r *http.Request) bool {
	if s.joinToken == "" {
		return true
	}
	return r.Header.Get("Authorization") == "Bearer "+s.joinToken
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

type errorCode string

const (
	errorUnauthorized          errorCode = "unauthorized"
	errorInvalidJSON           errorCode = "invalid_json"
	errorMissingNodeName       errorCode = "missing_node_name"
	errorMissingModel          errorCode = "missing_model"
	errorMissingMessages       errorCode = "missing_messages"
	errorUnknownModel          errorCode = "unknown_model"
	errorNoSingleNodeRoute     errorCode = "no_single_node_route"
	errorBackendConfigInvalid  errorCode = "backend_config_invalid"
	errorBackendRequestFailed  errorCode = "backend_request_failed"
	errorBenchmarkRecordFailed errorCode = "benchmark_record_failed"
	errorLayerSplitStageFailed errorCode = "layer_split_stage_failed"
	errorLayerSplitUnsupported errorCode = "layer_split_unsupported"
	errorNoLayerSplitRoute     errorCode = "no_layer_split_route"
)

func writeError(w http.ResponseWriter, status int, code errorCode, message string) {
	writeJSON(w, status, map[string]string{
		"error":   string(code),
		"message": message,
	})
}

func fallback(value string, defaultValue string) string {
	if strings.TrimSpace(value) == "" {
		return defaultValue
	}
	return value
}

func fallbackMap(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

func defaultBackendFactory(backend cluster.RuntimeBackend) (runtimeclient.ChatBackend, error) {
	return runtimeclient.NewOpenAIClient(backend.BaseURL, 60*time.Second)
}

func (s *Server) selectSingleNodeBackend(model cluster.ModelProfile) (cluster.NodeRecord, cluster.RuntimeBackend, error) {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)

	preview := routing.Preview(model, nodes)
	placementByNode := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placementByNode[placement.NodeName] = placement
	}
	for _, node := range nodes {
		placement := placementByNode[node.NodeName]
		if !placement.Valid {
			continue
		}
		for _, backend := range node.Backends {
			if backendCompatible(model, backend) {
				return node, backend, nil
			}
		}
	}
	if len(nodes) == 0 {
		return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no online nodes")
	}
	return cluster.NodeRecord{}, cluster.RuntimeBackend{}, fmt.Errorf("no compatible backend for model %q", model.ID)
}

func (s *Server) layerSplitCandidates(model cluster.ModelProfile) []layersplit.NodeCandidate {
	s.mu.RLock()
	nodes := make([]cluster.NodeRecord, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	s.mu.RUnlock()
	sortNodesByName(nodes)

	preview := routing.Preview(model, nodes)
	placementByNode := make(map[string]routing.PlacementPreview, len(preview.Placements))
	for _, placement := range preview.Placements {
		placementByNode[placement.NodeName] = placement
	}

	candidates := make([]layersplit.NodeCandidate, 0, len(nodes))
	for _, node := range nodes {
		placement := placementByNode[node.NodeName]
		if !placement.Valid {
			continue
		}
		backend, ok := firstCompatibleBackend(model, node.Backends)
		if !ok {
			continue
		}
		candidates = append(candidates, layersplit.NodeCandidate{
			NodeName:    node.NodeName,
			BackendID:   backend.ID,
			BackendKind: backend.Kind,
			BaseURL:     backend.BaseURL,
			Weight:      layerSplitWeight(node.Capabilities),
		})
	}
	return candidates
}

func firstCompatibleBackend(model cluster.ModelProfile, backends []cluster.RuntimeBackend) (cluster.RuntimeBackend, bool) {
	for _, backend := range backends {
		if backendCompatible(model, backend) {
			return backend, true
		}
	}
	return cluster.RuntimeBackend{}, false
}

func backendCompatible(model cluster.ModelProfile, backend cluster.RuntimeBackend) bool {
	if strings.TrimSpace(backend.BaseURL) == "" {
		return false
	}
	if !backend.OpenAICompatible {
		return false
	}
	if len(backend.Models) > 0 {
		for _, modelID := range backend.Models {
			if modelID == model.ID {
				return true
			}
		}
		return false
	}
	return backend.Kind == model.Runtime
}

func optionalFloat(values map[string]any, key string) *float64 {
	value, ok := values[key]
	if !ok {
		return nil
	}
	var output float64
	switch typed := value.(type) {
	case float64:
		output = typed
	case float32:
		output = float64(typed)
	case int:
		output = float64(typed)
	case int64:
		output = float64(typed)
	case json.Number:
		parsed, err := typed.Float64()
		if err != nil {
			return nil
		}
		output = parsed
	default:
		return nil
	}
	return &output
}

func layerSplitWeight(capabilities map[string]any) float64 {
	value := optionalFloat(capabilities, cluster.CapabilityLayerWeight)
	if value == nil || *value <= 0 {
		return 1
	}
	return *value
}

func sortNodesByName(nodes []cluster.NodeRecord) {
	slices.SortFunc(nodes, func(left cluster.NodeRecord, right cluster.NodeRecord) int {
		return cmp.Compare(left.NodeName, right.NodeName)
	})
}

func lastMessageContent(messages []chat.Message) string {
	for index := len(messages) - 1; index >= 0; index-- {
		content := strings.TrimSpace(messages[index].Content)
		if content != "" {
			return content
		}
	}
	return ""
}

func (s *Server) layerSplitRouteMetadata(plan layersplit.Plan, responses []layersplit.ActivationResponse, latency time.Duration) *chat.RouteMetadata {
	stages := make([]chat.RouteStage, 0, len(responses))
	bytesIn := 0
	bytesOut := 0
	for index, response := range responses {
		planStage := plan.Stages[index]
		bytesIn += response.BytesIn
		bytesOut += response.BytesOut
		stages = append(stages, chat.RouteStage{
			Index:       response.StageIndex,
			NodeName:    response.NodeName,
			BackendID:   planStage.BackendID,
			BackendKind: planStage.BackendKind,
			Role:        string(response.Role),
			LayerStart:  response.LayerStart,
			LayerEnd:    response.LayerEnd,
			Transport:   response.Transport,
			LatencyMS:   response.LatencyMS,
			BytesIn:     response.BytesIn,
			BytesOut:    response.BytesOut,
		})
	}
	metadata := &chat.RouteMetadata{
		Mode:      cluster.RouteModeLayerSplit,
		LatencyMS: latency.Milliseconds(),
		Stages:    stages,
		BytesIn:   bytesIn,
		BytesOut:  bytesOut,
	}
	if len(plan.Stages) > 0 {
		metadata.NodeName = plan.Stages[0].NodeName
		metadata.BackendID = plan.Stages[0].BackendID
		metadata.BackendKind = plan.Stages[0].BackendKind
	}
	return metadata
}

func stageNodeNames(responses []layersplit.ActivationResponse) []string {
	nodeNames := make([]string, 0, len(responses))
	for _, response := range responses {
		nodeNames = append(nodeNames, response.NodeName)
	}
	return nodeNames
}
