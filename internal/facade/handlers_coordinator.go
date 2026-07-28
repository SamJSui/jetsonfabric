package facade

import (
	"net/http"
	"net/http/httputil"
	"net/url"
)

func (r *Router) handleCoordinator(w http.ResponseWriter, req *http.Request) {
	leader, ok := r.leader()
	if !ok {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "leader_unavailable",
			"message": "no healthy leader-eligible node is currently known",
		})
		return
	}
	if leader.NodeID == r.selfID {
		r.serveLocalCoordinator(w, req)
		return
	}
	proxyToLeader(w, req, leader.APIURL)
}

func (r *Router) serveLocalCoordinator(w http.ResponseWriter, req *http.Request) {
	if r.coordinator == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error":   "coordinator_unavailable",
			"message": "this node is leader but has no coordinator router configured",
		})
		return
	}
	r.coordinator.ServeHTTP(w, req)
}

func proxyToLeader(w http.ResponseWriter, req *http.Request, leaderURL string) {
	target, err := url.Parse(leaderURL)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"error":   "invalid_leader_url",
			"message": err.Error(),
		})
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "leader_proxy_failed", "message": err.Error()})
	}
	proxy.ServeHTTP(w, req)
}
