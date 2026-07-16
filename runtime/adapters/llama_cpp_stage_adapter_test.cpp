#include "adapters/llama_cpp_adapter.hpp"
#include "adapters/llama_cpp_model.hpp"
#include "adapters/llama_cpp_stage_adapter.hpp"
#include "inference/stage.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using jetsonfabric::runtime::adapters::LlamaCppAdapter;
using jetsonfabric::runtime::adapters::LlamaCppModel;
using jetsonfabric::runtime::adapters::LlamaCppModelConfig;
using jetsonfabric::runtime::adapters::LlamaCppStageAdapter;
using jetsonfabric::runtime::adapters::LlamaCppStageConfig;
using namespace jetsonfabric::runtime::inference;

std::string model_path_from_environment() {
    const char* value = std::getenv("CI_MODEL_PATH");
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error("CI_MODEL_PATH is required");
    }
    return value;
}

Payload text_payload(const std::string& text) {
    Payload payload;
    payload.kind = PayloadKind::Text;
    payload.encoding = "utf-8";
    payload.bytes.assign(text.begin(), text.end());
    return payload;
}

StageInput input_for(
    const std::string& session_id,
    const std::string& request_id,
    Phase phase,
    int decode_step,
    StagePosition position,
    LayerRange layers,
    Payload payload
) {
    return StageInput{
        .session_id = session_id,
        .request_id = request_id,
        .model_id = "ci-model",
        .phase = phase,
        .decode_step = decode_step,
        .position = position,
        .layers = layers,
        .payload = std::move(payload),
        .max_tokens = 2,
    };
}

std::int32_t sampled_token(const Payload& payload) {
    if (payload.kind != PayloadKind::SampledToken || payload.bytes.size() != 4U) {
        throw std::runtime_error("expected one sampled token");
    }
    const std::uint32_t bits =
        static_cast<std::uint32_t>(payload.bytes[0]) |
        (static_cast<std::uint32_t>(payload.bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(payload.bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(payload.bytes[3]) << 24U);
    return static_cast<std::int32_t>(bits);
}

void require_success(const ExecutionResult& result, const char* step) {
    if (!result.ok) {
        throw std::runtime_error(std::string(step) + " failed: " + result.error.code + ": " + result.error.message);
    }
}

int run_test(const std::shared_ptr<LlamaCppModel>& model) {
    if (model->n_layer() < 2) {
        throw std::runtime_error("partial-layer test requires at least two transformer layers");
    }
    const int split = model->n_layer() / 2;
    const std::string prompt = "Once upon a time";

    LlamaCppAdapter baseline(model, 256, 2);
    const auto baseline_response = baseline.generate({.prompt = prompt, .max_tokens = 2});
    if (baseline_response.token_ids.empty()) {
        throw std::runtime_error("baseline did not generate a token");
    }

    LlamaCppStageAdapter stage0(LlamaCppStageConfig{
        .model = model,
        .ctx_size = 256,
        .threads = 2,
        .position = {.index = 0, .count = 2},
        .layers = {.start = 0, .end = split},
    });
    LlamaCppStageAdapter stage1(LlamaCppStageConfig{
        .model = model,
        .ctx_size = 256,
        .threads = 2,
        .position = {.index = 1, .count = 2},
        .layers = {.start = split, .end = model->n_layer()},
    });

    const std::string session = "native-split-test";
    ExecutionResult first_activation = stage0.execute(input_for(
        session, "prefill-0", Phase::Prefill, 0,
        {.index = 0, .count = 2}, {.start = 0, .end = split}, text_payload(prompt)
    ));
    require_success(first_activation, "stage 0 prefill");
    if (first_activation.output.payload.kind != PayloadKind::Activation ||
        first_activation.output.payload.tensor.dtype != "f32" ||
        first_activation.output.payload.tensor.shape.size() != 2 ||
        first_activation.output.payload.tensor.shape[1] != model->n_embd()) {
        throw std::runtime_error("stage 0 returned an invalid activation descriptor");
    }

    ExecutionResult first_token = stage1.execute(input_for(
        session, "prefill-1", Phase::Prefill, 0,
        {.index = 1, .count = 2}, {.start = split, .end = model->n_layer()},
        first_activation.output.payload
    ));
    require_success(first_token, "stage 1 prefill");
    if (sampled_token(first_token.output.payload) != baseline_response.token_ids[0]) {
        throw std::runtime_error("split prefill token does not match full-model greedy baseline");
    }

    if (baseline_response.token_ids.size() >= 2U) {
        ExecutionResult decode_activation = stage0.execute(input_for(
            session, "decode-0", Phase::Decode, 1,
            {.index = 0, .count = 2}, {.start = 0, .end = split}, first_token.output.payload
        ));
        require_success(decode_activation, "stage 0 decode");

        ExecutionResult second_token = stage1.execute(input_for(
            session, "decode-1", Phase::Decode, 1,
            {.index = 1, .count = 2}, {.start = split, .end = model->n_layer()},
            decode_activation.output.payload
        ));
        require_success(second_token, "stage 1 decode");
        if (sampled_token(second_token.output.payload) != baseline_response.token_ids[1]) {
            throw std::runtime_error("split decode token does not match full-model greedy baseline");
        }
    }

    if (stage0.session_count() != 1U || stage1.session_count() != 1U) {
        throw std::runtime_error("stage session contexts were not retained");
    }
    stage0.close_session(session);
    stage1.close_session(session);
    if (stage0.session_count() != 0U || stage1.session_count() != 0U) {
        throw std::runtime_error("stage session cleanup failed");
    }

    std::cout << "llama.cpp split-stage equivalence passed: architecture=" << model->architecture()
              << " layers=" << model->n_layer() << " split=" << split
              << " first_token=" << baseline_response.token_ids[0] << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = model_path_from_environment();
        auto model = std::make_shared<LlamaCppModel>(LlamaCppModelConfig{
            .model_path = model_path,
            .n_gpu_layers = 0,
        });
        if (argc == 2 && std::string(argv[1]) == "--print-layer-count") {
            std::cout << model->n_layer() << '\n';
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--baseline-token") {
            LlamaCppAdapter baseline(model, 256, 2);
            const auto response = baseline.generate({.prompt = "Once upon a time", .max_tokens = 1});
            if (response.token_ids.empty()) {
                throw std::runtime_error("baseline did not generate a token");
            }
            std::cout << response.token_ids[0] << '\n';
            return 0;
        }
        return run_test(model);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
