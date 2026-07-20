#include "engine/runtime_service.hpp"

#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"

#include <exception>
#include <sstream>
#include <utility>

namespace jetsonfabric::runtime {
namespace {

RuntimeResponse json_error(const std::string& status, const std::string& code, const std::string& message) {
    return RuntimeResponse{
        status,
        "application/json",
        "{\"error\":\"" + protocol::json_escape(code) + "\",\"message\":\"" +
            protocol::json_escape(message) + "\"}",
    };
}

protocol::StageResponse stage_error_response(
    const protocol::StageRequest& request,
    const std::string& code,
    const std::string& message
) {
    protocol::StageResponse response;
    response.session_id = request.session_id;
    response.request_id = request.request_id;
    response.model_id = request.model_id;
    response.phase = request.phase;
    response.decode_step = request.decode_step;
    response.stage_index = request.stage_index;
    response.stage_count = request.stage_count;
    response.node_name = request.node_name;
    response.layer_start = request.layer_start;
    response.layer_end = request.layer_end;
    response.payload_kind = request.payload_kind;
    response.encoding = request.encoding;
    response.dtype = request.dtype;
    response.shape = request.shape;
    response.byte_order = request.byte_order;
    response.layout = request.layout;
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    response.error = code;
    response.message = message;
    return response;
}

} // namespace

RuntimeService::RuntimeService(Config config)
    : config_(std::move(config)),
      model_manager_(config_) {}

std::string RuntimeService::runtime_name() const {
    return "jetsonfabric-runtime-worker";
}

std::string RuntimeService::engine_name() const {
    return config_.engine;
}

ExecutionMode RuntimeService::execution_mode() const {
    return config_.mode;
}

std::string RuntimeService::model() const {
    return model_manager_.active_model_id();
}

RuntimeResponse RuntimeService::deployment_status() const {
    const deployment::DeploymentStatus status = model_manager_.deployment_status();

    std::ostringstream body;
    body << "{\"resident\":" << (status.resident ? "true" : "false")
         << ",\"active\":" << (status.active ? "true" : "false");

    if (!status.resident) {
        body << ",\"state\":\"idle\",\"deployment\":null}";
        return RuntimeResponse{"200 OK", "application/json", body.str()};
    }

    if (!status.state.has_value() || !status.identity.has_value()) {
        return json_error(
            "500 Internal Server Error",
            "invalid_deployment_status",
            "resident deployment status is incomplete"
        );
    }

    body << ",\"state\":\""
         << deployment::resident_deployment_state_string(*status.state)
         << "\",\"deployment\":{\"deployment_id\":\""
         << protocol::json_escape(status.identity->deployment_id)
         << "\",\"model_id\":\""
         << protocol::json_escape(status.identity->model_id)
         << "\"}}";

    return RuntimeResponse{"200 OK", "application/json", body.str()};
}

RuntimeResponse RuntimeService::chat_completion(const std::string& /*request_body*/) const {
    return json_error(
        "501 Not Implemented",
        "chat_backend_not_implemented",
        "chat completions require an engine adapter; stage execution is implemented first"
    );
}

RuntimeResponse RuntimeService::run_stage(const std::string& request_body) const {
    if (config_.mode != ExecutionMode::PipelineParallel) {
        return json_error("400 Bad Request", "invalid_execution_mode", "stage execution requires pipeline_parallel mode");
    }

    std::string operation;
    protocol::StageRequest request;
    try {
        operation = protocol::decode_stage_operation(request_body);
        request = protocol::decode_stage_request(request_body);
    } catch (const std::exception& err) {
        return json_error("400 Bad Request", "invalid_stage_request", err.what());
    }

    const pipeline_parallel::StageRunResult result = operation == protocol::kStageOperationCloseSession
        ? model_manager_.close_session(request)
        : model_manager_.run_stage(request);
    if (!result.ok) {
        protocol::StageResponse response = stage_error_response(request, result.error_code, result.error_message);
        return RuntimeResponse{result.status, protocol::kStageWireContentType, protocol::encode_stage_response(std::move(response))};
    }

    return RuntimeResponse{
        "200 OK",
        protocol::kStageWireContentType,
        protocol::encode_stage_response(result.response),
    };
}

} // namespace jetsonfabric::runtime
