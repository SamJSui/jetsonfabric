package control

import "net/http"

type errorCode string

const (
	errorUnauthorized          errorCode = "unauthorized"
	errorInvalidJSON           errorCode = "invalid_json"
	errorMissingNodeName       errorCode = "missing_node_name"
	errorMissingModel          errorCode = "missing_model"
	errorMissingMessages       errorCode = "missing_messages"
	errorUnknownModel          errorCode = "unknown_model"
	errorNoSingleNodeRoute     errorCode = "no_single_node_route"
	errorBackendConfigInvalid  errorCode = "backend_config_invalid"
	errorBackendRequestFailed  errorCode = "backend_request_failed"
	errorBenchmarkRecordFailed errorCode = "benchmark_record_failed"
	errorLayerSplitStageFailed errorCode = "layer_split_stage_failed"
	errorLayerSplitUnsupported errorCode = "layer_split_unsupported"
	errorNoLayerSplitRoute     errorCode = "no_layer_split_route"
)

func writeError(w http.ResponseWriter, status int, code errorCode, message string) {
	writeJSON(w, status, map[string]string{
		"error":   string(code),
		"message": message,
	})
}
