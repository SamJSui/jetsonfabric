#include "engine/runtime_service.hpp"

#include "pipeline_parallel/generation_runner.hpp"
#include "protocol/execution_mode.hpp"
#include "protocol/generation.hpp"
#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"
#include "transport/http_stage_client.hpp"

#include <cstdint>
#include <cctype>
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

std::uint64_t require_positive_uint64(const nlohmann::json& body, const char* field) {
    const auto value = body.find(field);
    if (value == body.end() || value->is_null()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    if (value->is_number_unsigned()) {
        const std::uint64_t parsed = value->get<std::uint64_t>();
        if (parsed == 0) {
            throw std::invalid_argument(std::string(field) + " must be positive");
        }
        return parsed;
    }
    if (!value->is_number_integer()) {
        throw std::invalid_argument(std::string(field) + " must be a positive integer");
    }
    const std::int64_t parsed = value->get<std::int64_t>();
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return static_cast<std::uint64_t>(parsed);
}

std::string require_sha256(const nlohmann::json& body) {
    std::string value = require_string(body, "model_sha256");
    if (value.size() != 64) {
        throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
    }
    for (const unsigned char character : value) {
        if (!std::isxdigit(character)) {
            throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
        }
    }
    return value;
}

deployment::DeploymentIdentity decode_deployment_identity(const nlohmann::json& body) {
    return deployment::DeploymentIdentity{
        .deployment_id = require_string(body, "deployment_id"),
        .epoch = require_positive_uint64(body, "epoch"),
        .model_id = require_string(body, "model_id"),
        .model_sha256 = require_sha256(body),
    };
}

deployment::DeploymentIdentity decode_expected_deployment_identity(
    const std::string& request_body
) {
    return decode_deployment_identity(parse_request_object(request_body));
}

struct DecodedLoadRequest {
    deployment::DeploymentIdentity identity;
    Config config;
};

DecodedLoadRequest decode_load_request(const Config& base, const std::string& request_body) {
    const nlohmann::json body = parse_request_object(request_body);

    DecodedLoadRequest request;
    request.identity = decode_deployment_identity(body);
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

void append_model_residency(
    std::ostringstream& body,
    const deployment::DeploymentStatus& status
) {
    body << ",\"model_memory\":";
    if (!status.model_residency.has_value()) {
        body << "null";
        return;
    }
    const deployment::ModelResidency& memory = *status.model_residency;
    body << "{\"layer_start\":" << memory.layer_start
         << ",\"layer_end\":" << memory.layer_end
         << ",\"layer_count\":" << memory.layer_count
         << ",\"resident_weight_bytes\":" << memory.resident_weight_bytes
         << ",\"total_weight_bytes\":" << memory.total_weight_bytes
         << ",\"resident_tensor_count\":" << memory.resident_tensor_count
         << ",\"partitioned\":" << (memory.partitioned() ? "true" : "false")
         << ",\"pinned\":" << (status.active ? "true" : "false")
         << "}";
}

void append_deployment_identity(
    std::ostringstream& body,
    const deployment::DeploymentIdentity& identity
) {
    body << "{\"deployment_id\":\""
         << protocol::json_escape(identity.deployment_id)
         << "\",\"epoch\":" << identity.epoch
         << ",\"model_id\":\""
         << protocol::json_escape(identity.model_id)
         << "\",\"model_sha256\":\""
         << protocol::json_escape(identity.model_sha256)
         << "\"}";
}

RuntimeResponse operation_response(
    const char* operation,
    const deployment::DeploymentOperationResult& result,
    const deployment::DeploymentStatus& status
) {
    if (!result.identity.has_value()) {
        return json_error(
            "500 Internal Server Error",
            "invalid_deployment_result",
            "successful deployment operation omitted deployment identity"
        );
    }

    const std::string_view state = status.state.has_value()
        ? deployment::resident_deployment_state_string(*status.state)
        : "idle";
    std::ostringstream body;
    body << "{\"" << operation << "\":true,\"deployment\":";
    append_deployment_identity(body, *result.identity);
    body << ",\"resident\":" << (status.resident ? "true" : "false")
         << ",\"active\":" << (status.active ? "true" : "false")
         << ",\"state\":\"" << state << "\"";
    append_model_residency(body, status);
    body << "}";
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
    response.deployment_id = request.deployment_id;
    response.deployment_epoch = request.deployment_epoch;
    response.model_sha256 = request.model_sha256;
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
        body << ",\"state\":\"idle\",\"deployment\":null,\"model_memory\":null}";
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
         << "\",\"deployment\":";
    append_deployment_identity(body, *status.identity);
    append_model_residency(body, status);
    body << "}";

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
    return operation_response(
        "loaded",
        result,
        model_manager_.deployment_status(*result.identity)
    );
}

RuntimeResponse RuntimeService::activate_deployment(const std::string& request_body) {
    deployment::DeploymentIdentity expected_identity;
    try {
        expected_identity = decode_expected_deployment_identity(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_activate_request", err.what());
    }

    const deployment::ActivateDeploymentResult result =
        model_manager_.activate_resident_deployment(expected_identity);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    return operation_response(
        "activated",
        result,
        model_manager_.deployment_status(*result.identity)
    );
}

RuntimeResponse RuntimeService::drain_deployment(const std::string& request_body) {
    deployment::DeploymentIdentity expected_identity;
    try {
        expected_identity = decode_expected_deployment_identity(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_drain_request", err.what());
    }

    const deployment::DrainDeploymentResult result =
        model_manager_.drain_resident_deployment(expected_identity);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    return operation_response(
        "drained",
        result,
        model_manager_.deployment_status(*result.identity)
    );
}

RuntimeResponse RuntimeService::unload_deployment(const std::string& request_body) {
    deployment::DeploymentIdentity expected_identity;
    try {
        expected_identity = decode_expected_deployment_identity(request_body);
    } catch (const std::invalid_argument& err) {
        return json_error("400 Bad Request", "invalid_unload_request", err.what());
    }

    const deployment::UnloadDeploymentResult result =
        model_manager_.unload_resident_deployment(expected_identity);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }
    return operation_response(
        "unloaded",
        result,
        model_manager_.deployment_status(*result.identity)
    );
}

RuntimeResponse RuntimeService::chat_completion(const std::string& /*request_body*/) const {
    return json_error(
        "501 Not Implemented",
        "chat_backend_not_implemented",
        "chat completions require an engine adapter; stage execution is implemented first"
    );
}

RuntimeResponse RuntimeService::generate(
    const std::string& request_body,
    const GenerationEventSink& sink
) const {
    protocol::GenerationRequest request;
    try {
        request = protocol::decode_generation_request(request_body);
    } catch (const std::exception& error) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event("invalid_generation_request", error.what()),
        };
    }
    if (config_.mode != ExecutionMode::PipelineParallel) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                "invalid_execution_mode",
                "runtime-owned generation requires pipeline_parallel mode"
            ),
        };
    }

    const deployment::DeploymentIdentity* active = request.deployment.has_value()
        ? model_manager_.executable_deployment_identity(*request.deployment)
        : model_manager_.active_deployment_identity();
    if (active == nullptr) {
        const bool identity_mismatch = request.deployment.has_value() &&
            model_manager_.has_active_deployment();
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                identity_mismatch ? "deployment_mismatch" : "no_active_deployment",
                identity_mismatch
                    ? "generation deployment identity does not match an executable runtime epoch"
                    : "runtime has no executable deployment for the requested epoch"
            ),
        };
    }
    if (active->model_id != request.model_id) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                "deployment_mismatch",
                "generation model does not match the selected deployment"
            ),
        };
    }
    if (active->epoch > 0 && !request.deployment.has_value()) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                "deployment_identity_required",
                "managed runtime generation requires deployment identity"
            ),
        };
    }
    if (request.deployment.has_value() && *request.deployment != *active) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                "deployment_mismatch",
                "generation deployment identity does not match an executable runtime epoch"
            ),
        };
    }
    if (request.stages.front().stage_index != 0 ||
        request.stages.front().node_name != config_.node_name) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(
                "invalid_pipeline_leader",
                "generation must be sent to the runtime assigned stage zero"
            ),
        };
    }

    transport::HTTPStageClient peer_client(config_.cluster_token);
    pipeline_parallel::GenerationRunner runner([
        this,
        &peer_client
    ](
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& stage_request,
        pipeline_parallel::StageOperation operation
    ) {
        if (stage.stage_index == 0) {
            return operation == pipeline_parallel::StageOperation::CloseSession
                ? model_manager_.close_session(stage_request)
                : model_manager_.run_stage(stage_request);
        }
        return peer_client.invoke(stage, stage_request, operation);
    });
    const pipeline_parallel::GenerationResult result = runner.run(
        request,
        [&sink](const pipeline_parallel::GenerationToken& token) {
            return sink && sink(protocol::encode_generation_token_event(
                token.token,
                token.text,
                token.index
            ));
        }
    );
    if (!result.ok) {
        return RuntimeResponse{
            "200 OK",
            protocol::kGenerationContentType,
            protocol::encode_generation_error_event(result.error_code, result.error_message),
        };
    }
    return RuntimeResponse{
        "200 OK",
        protocol::kGenerationContentType,
        protocol::encode_generation_done_event(
            result.finish_reason,
            result.prompt_tokens,
            result.completion_tokens,
            result.sampled_tokens,
            result.stage_calls,
            result.remote_stage_calls,
            result.bytes_in,
            result.bytes_out
        ),
    };
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
