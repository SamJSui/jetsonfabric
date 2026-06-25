package api

import "net/http"

const (
	PathHealth          = "/healthz"
	PathNodes           = "/v1/nodes"
	PathModels          = "/v1/models"
	PathRoutePreview    = "/v1/routes/preview"
	PathAgentHeartbeat  = "/v1/agents/heartbeat"
	PathChatCompletions = "/v1/chat/completions"
)

const (
	RouteHealth          = http.MethodGet + " " + PathHealth
	RouteNodes           = http.MethodGet + " " + PathNodes
	RouteModels          = http.MethodGet + " " + PathModels
	RoutePreview         = http.MethodGet + " " + PathRoutePreview
	RouteAgentHeartbeat  = http.MethodPost + " " + PathAgentHeartbeat
	RouteChatCompletions = http.MethodPost + " " + PathChatCompletions
)
