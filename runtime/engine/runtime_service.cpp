#include "engine/runtime_service.hpp"

#include "protocol/execution_mode.hpp"
#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"

#include <cstdint>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

nlohmann::json parse_request_object(const std::string& request_body) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(request_body);
    } catch (const nlohmann::json::parse_error&) {
        throw std::invalid_argument("request body must be valid JSON");
    }
    if (!body.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    return body;
}

std::string require_string(const nlohmann::json& body, const char* field) {
    const auto value = body.find(field);
    if (value == body.end() || value->is_null()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    if (!value->is_string()) {
        throw std::invalid_argument(std::string(field) + " must be a string");
    }
    std::string parsed = value->get<std::string>();
    if (parsed.empty()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    return parsed;
}

std::string optional_string(
    const nlohmann::json& body,
    const char* field,
    const std::string& fallback
) {
    const auto value = body.find(field);
    if (value == body.end() || value->is_null()) {
        return fallback;
    }
    if (!value->is_string()) {
        throw std::invalid_argument(std::string(field) + " must be a string");
    }
    return value->get<std::string>();
}

int integer_value(const nlohmann::json& value, const char* field) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }
    try {
        const std::int64_t parsed = value.get<std::int64_t>();
        if (parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(std::string(field) + " is outside int range");
        }
        return static_cast<int>(parsed);
    } catch (const nlohmann::json::exception&) {
        throw std::invalid_argument(std::string(field) + " is outside int range");
    }
}

int require_int(const nlohmann::json& body, const char* field) {
    const auto value = body.find(field);
    if (value == body.end() || value->is_null()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    return integer_value(*value, field);
}

int optional_int(const nlohmann::json& body, const char* field, int fallback) {
    const auto value = body.find(field);
    return value == body.end() || value->is_null()
        ? fallback
        : integer_value(*value, field);
}

std::string decode_expected_deployment_id(const std::string& request_body) {
    return require_string(parse_request_object(request_body), "deployment_id");
}

struct DecodedLoadRequest {
    deployment::DeploymentIdentity identity;
    Config config;
};

DecodedLoadRequest decode_load_request(const Config& base, const std::string& request_body) {
    const nlohmann::json body = parse_request_object(request_body);

    DecodedLoadRequest request;
    request.identity = deployment::DeploymentIdentity{
        .deployment_id = require_string(body, "deployment_id"),
        .model_id = require_string(body, "model_id"),
    };
    request.config = base;
    request.config.start_idle = false;
    request.config.model = request.identity.model_id;
    request.config.engine = optional_string(body, "engine", request.config.engine);
    request.config.compute_backend = optional_string(
        body,
        "compute_backend",
        request.config.compute_backend
    );
    request.config.model_path = optional_string(body, "model_path", "");
    request.config.ctx_size = optional_int(body, "ctx_size", request.config.ctx_size);
    request.config.n_gpu_layers = optional_int(
        body,
        "n_gpu_layers",
        request.config.n_gpu_layers
    );
    request.config.threads = optional_int(body, "threads", request.config.threads);
    request.config.mode = parse_execution_mode(optional_string(
        body,
        "mode",
        std::string(execution_mode_string(request.config.mode))
    ));
    request.config.stage_assignment = pipeline_parallel::StageAssignment{
        .stage_index = require_int(body, "stage_index"),
        .stage_count = require_int(body, "stage_count"),
        .layer_start = require_int(body, "layer_start"),
        .layer_end = require_int(body, "layer_end"),
    };

    validate_deployment_config(request.config);
    return request;
}

RuntimeResponse operation_response(
    const char* operation,
    const deployment::DeploymentOperationResult& result,
    bool resident,
    bool active,
    std::string_view state
) {
    if (!result.identity.has_value()) {
        return json_error(
            "500 Internal Server Error",
            "invalid_deployment_result",
            "successful deployment operation omitted deployment identity"
        );
    }

    std::ostringstream body;
    body << "{\"" << operation << "\":true,\"deployment\":{\"deployment_id\":\""
         << protocol::json_escape(result.identity->deployment_id)
         << "\",\"model_id\":\""
         << protocol::json_escape(result.identity->model_id)
         << "\"},\"resident\":" << (resident ? "true" : "false")
         << ",\"active\":" << (active ? "true" : "false")
         << ",\"state\":\"" << state << "\"}";
    return RuntimeResponse{result.status, "application/json", body.str()};
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

RuntimeResponse RuntimeService::load_deployment(const std::string& request_body) {
    DecodedLoadRequest request;
    try {
        request = decode_load_request(config_, request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_load_request", err.what());
    }

    const Config deployment_config = request.config;
    deployment::LoadDeploymentResult result = model_manager_.load_resident_deployment(
        deployment_config.node_name,
        request.identity,
        deployment_config.stage_assignment,
        [deployment_config]() {
            return build_inference_engine_parts(deployment_config);
        }
    );
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }

    config_ = deployment_config;
    return operation_response("loaded", result, true, false, "ready");
}

RuntimeResponse RuntimeService::activate_deployment(const std::string& request_body) {
    std::string expected_deployment_id;
    try {
        expected_deployment_id = decode_expected_deployment_id(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_activate_request", err.what());
    }

    const deployment::ActivateDeploymentResult result =
        model_manager_.activate_resident_deployment(expected_deployment_id);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    return operation_response("activated", result, true, true, "active");
}

RuntimeResponse RuntimeService::unload_deployment(const std::string& request_body) {
    std::string expected_deployment_id;
    try {
        expected_deployment_id = decode_expected_deployment_id(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_unload_request", err.what());
    }

    const deployment::UnloadDeploymentResult result =
        model_manager_.unload_resident_deployment(expected_deployment_id);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    return operation_response("unloaded", result, false, false, "idle");
}

RuntimeResponse RuntimeService::chat_completion(const std::string& /*request_body*/) const {
    return json_error(
        "501 Not Implemented",
        "chat_backend_not_implemented",
        "chat completions require an engine adapter; stage execution is implemented first"
    );
}

RuntimeResponse RuntimeService::run_stage(const std::string& request_body) const {
    if (model_manager_.has_active_deployment() &&
        config_.mode != ExecutionMode::PipelineParallel) {
        return json_error(
            "400 Bad Request",
            "invalid_execution_mode",
            "stage execution requires pipeline_parallel mode"
        );
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
        return RuntimeResponse{
            result.status,
            protocol::kStageWireContentType,
            protocol::encode_stage_response(std::move(response))
        };
    }

    return RuntimeResponse{
        "200 OK",
        protocol::kStageWireContentType,
        protocol::encode_stage_response(result.response),
    };
}

} // namespace jetsonfabric::runtime
