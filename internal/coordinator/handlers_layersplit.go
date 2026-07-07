package coordinator

import (
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/cluster"
	"github.com/SamJSui/jetsonfabric/internal/layersplit"
)

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
	if !slices.Contains(model.PlacementModes, cluster.ExecutionModePipelineParallel) {
		writeError(w, http.StatusBadRequest, errorLayerSplitUnsupported, fmt.Sprintf("model %q does not allow pipeline_parallel placement", model.ID))
		return
	}

	candidates := s.layerSplitCandidates(model)
	plan, err := layersplit.PlanForModel(model, candidates)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, errorNoPipelineParallelRoute, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, plan)
}
