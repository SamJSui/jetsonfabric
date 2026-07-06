#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"

#include <chrono>
#include <exception>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

int elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
    );
}

StageRunResult error_result(const std::string& code, const std::string& message) {
    return StageRunResult{
        false,
        "502 Bad Gateway",
        code,
        message,
        {},
    };
}

adapters::GenerateRequest generate_request(const protocol::ActivationRequest& request) {
    return adapters::GenerateRequest{
        .prompt = request.payload,
        .max_tokens = request.max_tokens > 0 ? request.max_tokens : 128,
    };
}

void fill_trace(protocol::ActivationResponse& response) {
    response.trace.stage_index = response.stage_index;
    response.trace.node_name = response.node_name;
    response.trace.role = response.role;
    response.trace.layer_start = response.layer_start;
    response.trace.layer_end = response.layer_end;
    response.trace.bytes_in = response.bytes_in;
    response.trace.bytes_out = response.bytes_out;
    response.trace.transport = response.transport;
    response.trace.latency_ms = response.latency_ms;
}

protocol::ActivationResponse build_response(
    const protocol::ActivationRequest& request,
    const adapters::GenerateResponse& generated,
    int latency_ms
) {
    protocol::ActivationResponse response;
    response.session_id = request.session_id;
    response.request_id = request.request_id;
    response.model_id = request.model_id;
    response.stage_index = request.stage_index;
    response.node_name = request.node_name;
    response.role = request.role;
    response.layer_start = request.layer_start;
    response.layer_end = request.layer_end;
    response.decode_step = request.decode_step;
    response.shape = request.shape;
    response.dtype = request.dtype;
    response.payload = generated.text;
    response.bytes_in = request.bytes_in;
    response.bytes_out = static_cast<int>(generated.text.size());
    response.prompt_tokens = generated.prompt_tokens;
    response.completion_tokens = generated.completion_tokens;
    response.transport = request.transport;
    response.latency_ms = latency_ms;
    fill_trace(response);
    return response;
}

StageRunResult ok_result(protocol::ActivationResponse response) {
    return StageRunResult{
        true,
        "200 OK",
        "",
        "",
        response,
    };
}

} // namespace

LlamaCppFullModelExecutor::LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter)
    : adapter_(adapter) {}

StageRunResult LlamaCppFullModelExecutor::run_layers(const protocol::ActivationRequest& request) const {
    const auto start = std::chrono::steady_clock::now();
    adapters::GenerateResponse generated;

    try {
        generated = adapter_.generate(generate_request(request));
    } catch (const std::exception& err) {
        return error_result("generation_failed", err.what());
    }

    if (generated.text.empty()) {
        return error_result("empty_generation", "llama.cpp engine returned an empty generation");
    }

    protocol::ActivationResponse response = build_response(request, generated, elapsed_ms(start));
    return ok_result(response);
}

} // namespace jetsonfabric::runtime::pipeline_parallel
