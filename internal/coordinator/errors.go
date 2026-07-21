package coordinator

import "net/http"

type errorCode string

const (
	errorUnauthorized            errorCode = "unauthorized"
	errorInvalidJSON             errorCode = "invalid_json"
	errorMissingNodeName         errorCode = "missing_node_name"
	errorMissingModel            errorCode = "missing_model"
	errorMissingMessages         errorCode = "missing_messages"
	errorMissingPayload          errorCode = "missing_payload"
	errorInvalidStageCount       errorCode = "invalid_stage_count"
	errorUnknownModel            errorCode = "unknown_model"
	errorNoDataParallelRoute     errorCode = "no_single_node_route"
	errorBackendConfigInvalid    errorCode = "backend_config_invalid"
	errorBackendRequestFailed    errorCode = "backend_request_failed"
	errorBenchmarkRecordFailed   errorCode = "benchmark_record_failed"
	errorPipelineStageFailed     errorCode = "pipeline_parallel_stage_failed"
	errorLayerSplitUnsupported   errorCode = "pipeline_parallel_unsupported"
	errorNoPipelineParallelRoute errorCode = "no_pipeline_parallel_route"

	errorDeploymentConfigInvalid errorCode = "deployment_config_invalid"
	errorDeploymentUnavailable   errorCode = "deployment_unavailable"
	errorDeploymentPlanInvalid   errorCode = "deployment_plan_invalid"
	errorDeploymentTransitioning errorCode = "deployment_transitioning"
	errorDeploymentSwitchFailed  errorCode = "deployment_switch_failed"
	errorModelNotActive          errorCode = "model_not_active"
)

func writeError(w http.ResponseWriter, status int, code errorCode, message string) {
	writeJSON(w, status, map[string]string{
		"error":   string(code),
		"message": message,
	})
}
