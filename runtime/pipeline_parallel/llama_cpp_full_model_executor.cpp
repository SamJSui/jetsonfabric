#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"

#include <exception>
#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

adapters::GenerateRequest generate_request(const inference::StageInput& input) {
    return adapters::GenerateRequest{
        .prompt = std::string(input.payload.bytes.begin(), input.payload.bytes.end()),
        .max_tokens = input.max_tokens > 0 ? input.max_tokens : 128,
    };
}

inference::StageOutput text_output(const adapters::GenerateResponse& generated) {
    inference::StageOutput output;
    output.payload.kind = inference::PayloadKind::Text;
    output.payload.encoding = "utf-8";
    output.payload.bytes.assign(generated.text.begin(), generated.text.end());
    output.prompt_tokens = generated.prompt_tokens;
    output.completion_tokens = generated.completion_tokens;
    return output;
}

} // namespace

LlamaCppFullModelExecutor::LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter)
    : adapter_(adapter) {}

inference::ExecutionResult LlamaCppFullModelExecutor::execute(
    const inference::StageInput& input
) const {
    if (input.phase != inference::Phase::Prefill ||
        input.payload.kind != inference::PayloadKind::Text) {
        return inference::ExecutionResult::invalid_input(
            "full_model_text_only",
            "llama.cpp compatibility execution currently accepts prefill text only"
        );
    }

    adapters::GenerateResponse generated;
    try {
        generated = adapter_.generate(generate_request(input));
    } catch (const std::exception& error) {
        return inference::ExecutionResult::failure("generation_failed", error.what());
    }
    if (generated.text.empty()) {
        return inference::ExecutionResult::failure(
            "empty_generation",
            "llama.cpp engine returned an empty generation"
        );
    }
    return inference::ExecutionResult::success(text_output(generated));
}

} // namespace jetsonfabric::runtime::pipeline_parallel
