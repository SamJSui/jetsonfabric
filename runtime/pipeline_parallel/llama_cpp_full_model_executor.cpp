#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"

#include <chrono>
#include <exception>
#include <string>
#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

int elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

StageRunResult error_result(const std::string& code, const std::string& message) {
    return StageRunResult{false, "502 Bad Gateway", code, message, {}};
}

adapters::GenerateRequest generate_request(const protocol::StageRequest& request) {
    return adapters::GenerateRequest{
        .prompt = std::string(request.payload.begin(), request.payload.end()),
        .max_tokens = request.max_tokens > 0 ? request.max_tokens : 128,
    };
}

protocol::StageResponse build_response(
    const protocol::StageRequest& request,
    const adapters::GenerateResponse& generated,
    int latency_ms
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
    response.payload_kind = "text";
    response.encoding = "utf-8";
    response.payload.assign(generated.text.begin(), generated.text.end());
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    response.bytes_out = static_cast<std::int64_t>(response.payload.size());
    response.prompt_tokens = generated.prompt_tokens;
    response.completion_tokens = generated.completion_tokens;
    response.latency_ms = latency_ms;
    return response;
}

StageRunResult ok_result(protocol::StageResponse response) {
    return StageRunResult{true, "200 OK", "", "", std::move(response)};
}

} // namespace

LlamaCppFullModelExecutor::LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter)
    : adapter_(adapter) {}

StageRunResult LlamaCppFullModelExecutor::run_layers(const protocol::StageRequest& request) const {
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
    return ok_result(build_response(request, generated, elapsed_ms(start)));
}

} // namespace jetsonfabric::runtime::pipeline_parallel
