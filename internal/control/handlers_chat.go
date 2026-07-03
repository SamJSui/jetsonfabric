package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/benchmarks"
	"github.com/SamJSui/jetsonfabric/internal/chat"
	"github.com/SamJSui/jetsonfabric/internal/cluster"
)

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
	node, engine, err := s.selectDataParallelEngine(model)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoSingleNodeRoute, err.Error())
		return
	}

	chatBackend, err := s.engineFactory(engine)
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
		Mode:             cluster.ExecutionModeDataParallel,
		NodeName:         node.NodeName,
		Engine:           engine.Engine,
		EngineInstanceID: engine.InstanceID,
		LatencyMS:        stats.Latency.Milliseconds(),
	}

	record := benchmarks.Record{
		Timestamp:        s.now(),
		ModelID:          model.ID,
		NodeName:         node.NodeName,
		ExecutionMode:    cluster.ExecutionModeDataParallel,
		Engine:           engine.Engine,
		EngineInstanceID: engine.InstanceID,
		LatencyMS:        stats.Latency.Milliseconds(),
		OutputTokens:     stats.OutputTokens,
		TokensPerSec:     stats.TokensPerSec,
		MemoryGB:         optionalFloat(node.Capabilities, cluster.CapabilityMemoryGB),
		TemperatureC:     optionalFloat(node.Metrics, cluster.MetricTemperatureC),
	}
	if err := s.benchmarkRecorder.Record(r.Context(), record); err != nil {
		writeError(w, http.StatusInternalServerError, errorBenchmarkRecordFailed, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}
