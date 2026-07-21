package api

import "net/http"

const (
	HeaderCoordinatorNodeID = "X-JetsonFabric-Coordinator-Node-ID"
	HeaderClusterToken      = "X-JetsonFabric-Cluster-Token"
)

const (
	PathHealth                    = "/healthz"
	PathModels                    = "/v1/models"
	PathRoutePreview              = "/v1/routes/preview"
	PathLayerSplitRun             = "/v1/layer-split/run"
	PathLayerSplitStage           = "/v1/layer-split/stage"
	PathChatCompletions           = "/v1/chat/completions"
	PathDeploymentStatus          = "/v1/deployments/active"
	PathDeploymentSwitch          = "/v1/deployments/switch"
	PathRuntimeDeploymentStatus   = "/v1/runtime/deployment"
	PathRuntimeDeploymentLoad     = "/v1/runtime/deployment/load"
	PathRuntimeDeploymentActivate = "/v1/runtime/deployment/activate"
	PathRuntimeDeploymentUnload   = "/v1/runtime/deployment/unload"
	PathRuntimeGeneration         = "/v1/runtime/generate"
)

const (
	RouteHealth                    = http.MethodGet + " " + PathHealth
	RouteModels                    = http.MethodGet + " " + PathModels
	RoutePreview                   = http.MethodGet + " " + PathRoutePreview
	RouteLayerSplitRun             = http.MethodPost + " " + PathLayerSplitRun
	RouteLayerSplitStage           = http.MethodPost + " " + PathLayerSplitStage
	RouteChatCompletions           = http.MethodPost + " " + PathChatCompletions
	RouteDeploymentStatus          = http.MethodGet + " " + PathDeploymentStatus
	RouteDeploymentSwitch          = http.MethodPost + " " + PathDeploymentSwitch
	RouteRuntimeDeploymentStatus   = http.MethodGet + " " + PathRuntimeDeploymentStatus
	RouteRuntimeDeploymentLoad     = http.MethodPost + " " + PathRuntimeDeploymentLoad
	RouteRuntimeDeploymentActivate = http.MethodPost + " " + PathRuntimeDeploymentActivate
	RouteRuntimeDeploymentUnload   = http.MethodPost + " " + PathRuntimeDeploymentUnload
	RouteRuntimeGeneration         = http.MethodPost + " " + PathRuntimeGeneration
)
