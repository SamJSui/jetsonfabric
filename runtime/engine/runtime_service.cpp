#include "engine/runtime_service.hpp"

#include "activation/activation_codec_factory.hpp"
#include "engine/generation_service.hpp"
#include "speculative/draft_strategy_factory.hpp"
#include "protocol/execution_mode.hpp"
#include "protocol/generation.hpp"
#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"
#include "transport/stage_transport_factory.hpp"

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
    request.config.model_sha256 = request.identity.model_sha256;
    request.config.engine = optional_string(body, "engine", request.config.engine);
    const std::string activation_encoding = optional_string(
        body,
        "activation_encoding",
        request.config.activation_encoding
    );
    if (activation_encoding != request.config.activation_encoding) {
        throw std::invalid_argument(
            "activation_encoding does not match the runtime activation codec"
        );
    }
    const std::string kv_cache_type = optional_string(
        body,
        "kv_cache_type",
        std::string(kv_cache_type_string(request.config.kv_cache_type))
    );
    if (kv_cache_type != kv_cache_type_string(request.config.kv_cache_type)) {
        throw std::invalid_argument(
            "kv_cache_type does not match the runtime KV cache configuration"
        );
    }
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

const InferenceEngineFactory& require_engine_factory(
    const std::shared_ptr<const InferenceEngineFactory>& factory
) {
    if (!factory) {
        throw std::invalid_argument("runtime service requires an inference engine factory");
    }
    return *factory;
}

} // namespace

RuntimeService::RuntimeService(Config config)
    : RuntimeService(
          config,
          make_default_inference_engine_factory(),
          transport::make_default_stage_transport_factory()->create_transport(config)
      ) {}

RuntimeService::RuntimeService(
    Config config,
    std::shared_ptr<const InferenceEngineFactory> engine_factory,
    std::shared_ptr<const transport::StageTransport> stage_transport
)
    : RuntimeService(
          config,
          std::move(engine_factory),
          std::move(stage_transport),
          activation::make_default_activation_codec_factory()->create_codec(
              config.activation_encoding
          )
      ) {}

RuntimeService::RuntimeService(
    Config config,
    std::shared_ptr<const InferenceEngineFactory> engine_factory,
    std::shared_ptr<const transport::StageTransport> stage_transport,
    std::shared_ptr<const activation::ActivationCodec> activation_codec
)
    : config_(std::move(config)),
      engine_factory_(std::move(engine_factory)),
      stage_transport_(std::move(stage_transport)),
      activation_codec_(std::move(activation_codec)),
      model_manager_(
          config_,
          require_engine_factory(engine_factory_),
          activation_codec_
      ) {
    if (!stage_transport_) {
        throw std::invalid_argument("runtime service requires a stage transport");
    }
    if (!engine_factory_->supports(config_.engine)) {
        throw std::invalid_argument("unsupported inference engine: " + config_.engine);
    }
    if (!activation_codec_) {
        throw std::invalid_argument("runtime service requires an activation codec");
    }
}

void RuntimeService::shutdown() {
    stage_transport_->shutdown();
}

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

std::string RuntimeService::stage_transport_name() const {
    return config_.stage_transport;
}

std::string RuntimeService::activation_encoding() const {
    return config_.activation_encoding;
}

std::string RuntimeService::kv_cache_type() const {
    return kv_cache_type_string(config_.kv_cache_type);
}

int RuntimeService::ubatch_size() const {
    return config_.ubatch_size;
}

int RuntimeService::parallel_sessions() const {
    return config_.parallel_sessions;
}

int RuntimeService::decode_batch_size() const {
    return config_.decode_batch_size;
}

std::string RuntimeService::speculative_draft() const {
    return config_.speculative_draft;
}

int RuntimeService::speculative_max_tokens() const {
    return config_.speculative_max_tokens;
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
    if (const auto duplicate = model_manager_.resident_load_conflict(request.identity);
        duplicate.has_value()) {
        return json_error(
            duplicate->status,
            duplicate->error_code,
            duplicate->error_message
        );
    }

    const std::lock_guard load_lock(deployment_load_mutex_);
    if (const auto duplicate = model_manager_.resident_load_conflict(request.identity);
        duplicate.has_value()) {
        return json_error(
            duplicate->status,
            duplicate->error_code,
            duplicate->error_message
        );
    } else {
        std::optional<deployment::LoadMemoryEstimate> estimate;
        try {
            estimate = engine_factory_->estimate_load_memory(deployment_config);
        } catch (const std::invalid_argument& err) {
            return json_error("400 Bad Request", "invalid_engine_config", err.what());
        }
        const deployment::MemoryAdmissionDecision admission =
            deployment::assess_load_memory(
                deployment::available_memory_bytes(),
                estimate
            );
        if (!admission.admitted) {
            return json_error(
                "503 Service Unavailable",
                "deployment_memory_admission_rejected",
                admission.rejection_message()
            );
        }
    }
    deployment::LoadDeploymentResult result = model_manager_.load_resident_deployment(
        deployment_config.node_name,
        request.identity,
        deployment_config.stage_assignment,
        [this, deployment_config]() {
            return engine_factory_->create_engine(deployment_config);
        }
    );
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }

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
    const GenerationService generation_service(
        config_.node_name,
        config_.mode,
        model_manager_,
        *stage_transport_,
        speculative::create_draft_strategy(config_.speculative_draft),
        config_.speculative_max_tokens
    );
    const pipeline_parallel::GenerationResult result = generation_service.generate(
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
            result.rollback_stage_calls,
            result.remote_rollback_stage_calls,
            result.rollback_stage_us,
            result.rollback_remote_call_us,
            result.bytes_in,
            result.bytes_out,
            result.stage_timings,
            result.target_decode_passes,
            result.speculative_draft_tokens,
            result.speculative_accepted_tokens
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
        protocol::validate_stage_operation(operation, request.rollback_tokens);
    } catch (const std::exception& err) {
        return json_error("400 Bad Request", "invalid_stage_request", err.what());
    }

    protocol::StageResponse error_response = stage_error_response(request, "", "");
    pipeline_parallel::StageRunResult result = operation == protocol::kStageOperationCloseSession
        ? model_manager_.close_session(request)
        : operation == protocol::kStageOperationRollbackSession
            ? model_manager_.rollback_session(request)
            : model_manager_.run_stage(std::move(request));
    if (!result.ok) {
        error_response.operation = operation;
        error_response.error = result.error_code;
        error_response.message = result.error_message;
        protocol::EncodedStageFrame frame =
            protocol::encode_stage_response_frame(std::move(error_response));
        return RuntimeResponse{
            result.status,
            protocol::kStageWireContentType,
            std::move(frame.prefix),
            std::move(frame.payload),
        };
    }

    protocol::StageResponse response = std::move(result.response);
    response.operation = operation;
    protocol::EncodedStageFrame frame =
        protocol::encode_stage_response_frame(std::move(response));
    return RuntimeResponse{
        "200 OK",
        protocol::kStageWireContentType,
        std::move(frame.prefix),
        std::move(frame.payload),
    };
}

} // namespace jetsonfabric::runtime
