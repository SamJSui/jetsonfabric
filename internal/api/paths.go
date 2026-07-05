package api

import "net/http"

const (
	PathHealth = "/healthz"
	PathModels = "/v1/models"
	PathRoutePreview = "/v1/routes/preview"
	PathLayerSplitPlan = "/v1/layer-split/plan"
	PathLayerSplitStage = "/v1/layer-split/stage"
	PathLayerSplitChat = "/v1/layer-split/completions"
	PathChatCompletions = "/v1/chat/completions"
)

const (
	RouteHealth = http.MethodGet + " " + PathHealth
	RouteModels = http.MethodGet + " " + PathModels
	RoutePreview = http.MethodGet + " " + PathRoutePreview
	RouteLayerSplitPlan = http.MethodGet + " " + PathLayerSplitPlan
	RouteLayerSplitStage = http.MethodPost + " " + PathLayerSplitStage
	RouteLayerSplitChat = http.MethodPost + " " + PathLayerSplitChat
	RouteChatCompletions = http.MethodPost + " " + PathChatCompletions
)
