package facade

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/SamJSui/jetsonfabric/internal/api"
)

func (r *Router) handleStageRun(w http.ResponseWriter, req *http.Request) {
	if r.stageRunner == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "runtime_stage_gateway_unavailable",
			"message": "this node has no runtime stage gateway configured",
		})
		return
	}
	if !r.authorizeClusterToken(w, req, "runtime stage requests") {
		return
	}
	r.stageRunner.ServeHTTP(w, req)
}

func (r *Router) handleRuntimeDeployment(w http.ResponseWriter, req *http.Request) {
	if r.runtimeDeployment == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "runtime_deployment_gateway_unavailable",
			"message": "this node has no runtime deployment gateway configured",
		})
		return
	}
	if req.Method != http.MethodGet {
		if !r.authorizeCoordinatorRequest(w, req, "runtime lifecycle writes") {
			return
		}
	}
	r.runtimeDeployment.ServeHTTP(w, req)
}

func (r *Router) handleRuntimeGeneration(w http.ResponseWriter, req *http.Request) {
	if r.runtimeGeneration == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "runtime_generation_gateway_unavailable",
			"message": "this node has no runtime generation gateway configured",
		})
		return
	}
	if !r.authorizeCoordinatorRequest(w, req, "runtime generation requests") {
		return
	}
	r.runtimeGeneration.ServeHTTP(w, req)
}

func (r *Router) authorizeCoordinatorRequest(w http.ResponseWriter, req *http.Request, operation string) bool {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "coordinator_unavailable",
			"message": operation + " require an elected coordinator",
		})
		return false
	}
	if r.clusterToken == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "coordinator_auth_unconfigured",
			"message": operation + " require JETSONFABRIC_CLUSTER_TOKEN on every node",
		})
		return false
	}
	providedToken := req.Header.Get(api.HeaderClusterToken)
	if strings.TrimSpace(req.Header.Get(api.HeaderCoordinatorNodeID)) != leader.NodeID ||
		subtle.ConstantTimeCompare([]byte(providedToken), []byte(r.clusterToken)) != 1 {
		writeJSON(w, http.StatusForbidden, map[string]string{
			"error":   "coordinator_authentication_required",
			"message": operation + " require the elected coordinator and cluster token",
		})
		return false
	}
	return true
}

func (r *Router) authorizeClusterToken(w http.ResponseWriter, req *http.Request, operation string) bool {
	if r.clusterToken == "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "cluster_auth_unconfigured",
			"message": operation + " require JETSONFABRIC_CLUSTER_TOKEN on every node",
		})
		return false
	}
	providedToken := req.Header.Get(api.HeaderClusterToken)
	if subtle.ConstantTimeCompare([]byte(providedToken), []byte(r.clusterToken)) != 1 {
		writeJSON(w, http.StatusForbidden, map[string]string{
			"error":   "cluster_authentication_required",
			"message": operation + " require the cluster token",
		})
		return false
	}
	return true
}
