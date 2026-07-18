package coordinator

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/clusterplan"
	"github.com/SamJSui/jetsonfabric/internal/membership"
)

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.registry)
}

func (s *Server) handleRoutePreview(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("model")
	model, ok := s.registry.Find(modelID)
	if !ok {
		writeJSON(w, http.StatusOK, clusterplan.RoutePreview{
			Model:  modelID,
			Reason: clusterplan.Reason("unknown_model"),
		})
		return
	}

	var members []membership.Member
	if s.memberSource != nil {
		members = s.memberSource.List()
	}
	writeJSON(w, http.StatusOK, clusterplan.PreviewPipeline(clusterplan.Request{
		Model:      model,
		Members:    members,
		Now:        s.now(),
		StaleAfter: s.memberStaleAfter,
		Policy:     s.routePreviewPolicy(r),
	}))
}

func (s *Server) routePreviewPolicy(r *http.Request) clusterplan.Policy {
	policy := s.clusterPlanPolicy
	if queryBool(r, "allow_colocated_stages") {
		policy.AllowColocatedStages = true
	}
	if stageCount, present := queryOptionalPositiveInt(r, "stage_count"); present {
		policy.StageCount = stageCount
	}
	return policy
}

func queryBool(r *http.Request, key string) bool {
	value := strings.TrimSpace(r.URL.Query().Get(key))
	if value == "" {
		return false
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false
	}
	return parsed
}

func queryOptionalPositiveInt(r *http.Request, key string) (int, bool) {
	if _, present := r.URL.Query()[key]; !present {
		return 0, false
	}
	value := strings.TrimSpace(r.URL.Query().Get(key))
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return -1, true
	}
	return parsed, true
}
