#include "engine/runtime_service.hpp"

#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"

#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

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

std::string decode_unload_deployment_id(const std::string& request_body) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(request_body);
    } catch (const nlohmann::json::parse_error&) {
        throw std::invalid_argument("request body must be valid JSON");
    }
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }

    const auto deployment_id = body.find("deployment_id");
    if (deployment_id == body.end() || deployment_id->is_null()) {
        throw std::invalid_argument("deployment_id is required");
    }
    if (!deployment_id->is_string()) {
        throw std::invalid_argument("deployment_id must be a string");
    }

    std::string value = deployment_id->get<std::string>();
    if (value.empty()) {
        throw std::invalid_argument("deployment_id is required");
    }
    return value;
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

RuntimeResponse RuntimeService::unload_deployment(const std::string& request_body) {
    std::string expected_deployment_id;
    try {
        expected_deployment_id = decode_unload_deployment_id(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_unload_request", err.what());
    }

    deployment::UnloadDeploymentResult result =
        model_manager_.unload_resident_deployment(expected_deployment_id);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    if (!result.identity.has_value()) {
        return json_error(
            "500 Internal Server Error",
            "invalid_unload_result",
            "successful unload omitted deployment identity"
        );
    }

    std::ostringstream body;
    body << "{\"unloaded\":true,\"deployment\":{\"deployment_id\":\""
         << protocol::json_escape(result.identity->deployment_id)
         << "\",\"model_id\":\""
         << protocol::json_escape(result.identity->model_id)
         << "\"},\"resident\":false,\"active\":false,\"state\":\"idle\"}";
    return RuntimeResponse{result.status, "application/json", body.str()};
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