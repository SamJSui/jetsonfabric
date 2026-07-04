#include "engine/runtime_engine.hpp"

#include "protocol/activation.hpp"

#include <exception>
#include <utility>

namespace jetsonfabric::runtime {
namespace {

EngineResponse json_error(const std::string& status, const std::string& code, const std::string& message) {
    return EngineResponse{
        status,
        "{"
        "\"error\":\"" + protocol::json_escape(code) + "\","
        "\"message\":\"" + protocol::json_escape(message) + "\""
        "}",
    };
}

} // namespace

RuntimeEngine::RuntimeEngine(Config config)
    : config_(std::move(config)),
      layer_executor_(),
      stage_worker_(config_.node_name, config_.stage_assignment, layer_executor_) {}

std::string RuntimeEngine::runtime_name() const {
    return "jetsonfabric-runtime";
}

ExecutionMode RuntimeEngine::mode() const {
    return config_.mode;
}

std::string RuntimeEngine::model() const {
    return config_.model;
}

EngineResponse RuntimeEngine::chat_completion(const std::string& /*request_body*/) const {
    return json_error(
        "501 Not Implemented",
        "chat_backend_not_implemented",
        "chat completions require a model backend; runtime pipeline stage execution is being implemented first"
    );
}

EngineResponse RuntimeEngine::run_stage(const std::string& request_body) const {
    if (config_.mode != ExecutionMode::PipelineParallel) {
        return json_error(
            "400 Bad Request",
            "invalid_execution_mode",
            "stage execution requires runtime mode pipeline_parallel"
        );
    }

    protocol::ActivationRequest request;
    try {
        request = protocol::decode_activation_request(request_body);
    } catch (const std::exception& err) {
        return json_error("400 Bad Request", "invalid_activation_request", err.what());
    }

    const pipeline_parallel::StageRunResult result = stage_worker_.run(request);
    if (!result.ok) {
        return json_error(result.status, result.error_code, result.error_message);
    }

    return EngineResponse{
        "200 OK",
        protocol::encode_activation_response(result.response),
    };
}

} // namespace jetsonfabric::runtime