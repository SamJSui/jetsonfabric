#include "engine/runtime_service.hpp"

#include "engine/inference_engine_factory.hpp"
#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"

#include <exception>
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
      engine_parts_(build_inference_engine_parts(config_)),
      stage_worker_(config_.node_name, config_.model, config_.stage_assignment, *engine_parts_.layer_executor) {}

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
    return config_.model;
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
        ? stage_worker_.close_session(request)
        : stage_worker_.run(request);
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
