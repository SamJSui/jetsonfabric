package api

import "net/http"

const (
	PathHealth          = "/healthz"
	PathModels          = "/v1/models"
	PathRoutePreview    = "/v1/routes/preview"
	PathLayerSplitRun   = "/v1/layer-split/run"
	PathLayerSplitStage = "/v1/layer-split/stage"
	PathChatCompletions = "/v1/chat/completions"
)

const (
	RouteHealth          = http.MethodGet + " " + PathHealth
	RouteModels          = http.MethodGet + " " + PathModels
	RoutePreview         = http.MethodGet + " " + PathRoutePreview
	RouteLayerSplitRun   = http.MethodPost + " " + PathLayerSplitRun
	RouteLayerSplitStage = http.MethodPost + " " + PathLayerSplitStage
	RouteChatCompletions = http.MethodPost + " " + PathChatCompletions
)
