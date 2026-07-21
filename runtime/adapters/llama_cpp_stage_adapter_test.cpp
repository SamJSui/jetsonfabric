#include "adapters/llama_cpp_adapter.hpp"
#include "adapters/llama_cpp_model.hpp"
#include "adapters/llama_cpp_stage_adapter.hpp"
#include "inference/stage.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using jetsonfabric::runtime::adapters::LlamaCppAdapter;
using jetsonfabric::runtime::adapters::LlamaCppModel;
using jetsonfabric::runtime::adapters::LlamaCppModelConfig;
using jetsonfabric::runtime::adapters::LlamaCppStageAdapter;
using jetsonfabric::runtime::adapters::LlamaCppStageConfig;
using namespace jetsonfabric::runtime::inference;

constexpr int integration_max_tokens = 4;

std::string model_path_from_environment() {
    const char* value = std::getenv("CI_MODEL_PATH");
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error("CI_MODEL_PATH is required");
    }
    return value;
}

std::string baseline_prompt_from_environment() {
    const char* value = std::getenv("JF_BASELINE_PROMPT");
    return value == nullptr || *value == '\0' ? "Once upon a time" : value;
}

int baseline_max_tokens_from_environment() {
    const char* value = std::getenv("JF_BASELINE_MAX_TOKENS");
    if (value == nullptr || *value == '\0') {
        return integration_max_tokens;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 256) {
        throw std::runtime_error("JF_BASELINE_MAX_TOKENS must be an integer between 1 and 256");
    }
    return static_cast<int>(parsed);
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
        .max_tokens = integration_max_tokens,
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

void print_token_ids(const std::vector<std::int32_t>& token_ids) {
    std::cout << '[';
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << token_ids[index];
    }
    std::cout << "]\n";
}

void verify_single_stage_equivalence(
    const std::shared_ptr<LlamaCppModel>& model,
    const std::string& prompt,
    const std::vector<std::int32_t>& baseline_tokens
) {
    LlamaCppStageAdapter stage(LlamaCppStageConfig{
        .model = model,
        .ctx_size = 256,
        .threads = 2,
        .position = {.index = 0, .count = 1},
        .layers = {.start = 0, .end = model->n_layer()},
    });

    const std::string session = "native-single-stage-test";
    ExecutionResult result = stage.execute(input_for(
        session, "single-prefill", Phase::Prefill, 0,
        {.index = 0, .count = 1}, {.start = 0, .end = model->n_layer()}, text_payload(prompt)
    ));
    require_success(result, "single stage prefill");
    if (sampled_token(result.output.payload) != baseline_tokens[0]) {
        throw std::runtime_error("single-stage prefill token does not match full-model greedy baseline");
    }

    Payload payload = std::move(result.output.payload);
    for (std::size_t token_index = 1; token_index < baseline_tokens.size(); ++token_index) {
        const std::string request_id = "single-decode-" + std::to_string(token_index);
        result = stage.execute(input_for(
            session, request_id, Phase::Decode, static_cast<int>(token_index),
            {.index = 0, .count = 1}, {.start = 0, .end = model->n_layer()}, std::move(payload)
        ));
        require_success(result, request_id.c_str());
        if (sampled_token(result.output.payload) != baseline_tokens[token_index]) {
            throw std::runtime_error("single-stage decode token does not match full-model greedy baseline");
        }
        payload = std::move(result.output.payload);
    }

    if (stage.session_count() != 1U) {
        throw std::runtime_error("single-stage session context was not retained");
    }
    stage.close_session(session);
    if (stage.session_count() != 0U) {
        throw std::runtime_error("single-stage session cleanup failed");
    }
}

void verify_idle_session_expiry(const std::shared_ptr<LlamaCppModel>& model, int split) {
    LlamaCppStageAdapter expiring_stage(LlamaCppStageConfig{
        .model = model,
        .ctx_size = 256,
        .threads = 2,
        .position = {.index = 0, .count = 2},
        .layers = {.start = 0, .end = split},
        .session_idle_ttl = std::chrono::milliseconds(30),
        .session_reap_interval = std::chrono::milliseconds(5),
    });
    const std::string session = "expiring-session";
    ExecutionResult prefill = expiring_stage.execute(input_for(
        session, "expiring-prefill", Phase::Prefill, 0,
        {.index = 0, .count = 2}, {.start = 0, .end = split}, text_payload("TTL test")
    ));
    require_success(prefill, "expiring stage prefill");
    if (expiring_stage.session_count() != 1U) {
        throw std::runtime_error("expiring stage did not retain its new session");
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (expiring_stage.session_count() != 0U && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (expiring_stage.session_count() != 0U) {
        throw std::runtime_error("idle stage session was not reaped after its TTL");
    }
}

std::vector<LayerRange> balanced_layer_ranges(int layer_count, int stage_count) {
    std::vector<LayerRange> ranges;
    ranges.reserve(static_cast<std::size_t>(stage_count));
    const int width = layer_count / stage_count;
    const int remainder = layer_count % stage_count;
    int start = 0;
    for (int index = 0; index < stage_count; ++index) {
        const int end = start + width + (index < remainder ? 1 : 0);
        ranges.push_back({.start = start, .end = end});
        start = end;
    }
    return ranges;
}

void verify_partition_residency(
    const std::shared_ptr<LlamaCppModel>& full_model,
    const std::vector<std::shared_ptr<LlamaCppModel>>& stage_models,
    const std::vector<LayerRange>& ranges
) {
    const std::uint64_t full_bytes = full_model->resident_weight_bytes();
    if (full_bytes == 0) {
        throw std::runtime_error("model residency accounting returned zero bytes");
    }
    std::uint64_t per_device_capacity = 0;
    for (std::size_t index = 0; index < stage_models.size(); ++index) {
        const auto& stage_model = stage_models[index];
        const LayerRange range = ranges[index];
        if (stage_model->loaded_layer_start() != range.start ||
            stage_model->loaded_layer_end() != range.end) {
            throw std::runtime_error("partial model did not retain its assigned layer range");
        }
        const std::uint64_t stage_bytes = stage_model->resident_weight_bytes();
        if (stage_bytes == 0 || stage_bytes >= full_bytes) {
            throw std::runtime_error("a stage partition is not smaller than the full-model residency");
        }
        if (stage_model->total_weight_bytes() != full_bytes) {
            throw std::runtime_error("partial model did not retain full-model weight accounting");
        }
        if (stage_model->resident_tensor_count() >= full_model->resident_tensor_count()) {
            throw std::runtime_error("a stage partition retained every full-model tensor");
        }
        per_device_capacity = std::max(per_device_capacity, stage_bytes);
    }
    if (full_bytes <= per_device_capacity) {
        throw std::runtime_error("partition capacity proof did not exclude the full model");
    }
    std::cout << "model residency: full_bytes=" << full_bytes
              << " stages=" << stage_models.size()
              << " per_device_capacity=" << per_device_capacity << '\n';
}

void verify_pipeline_equivalence(
    const std::shared_ptr<LlamaCppModel>& model,
    const std::string& model_path,
    const std::string& prompt,
    const std::vector<std::int32_t>& baseline_tokens,
    int stage_count
) {
    if (model->n_layer() < stage_count) {
        throw std::runtime_error("model has fewer layers than the requested real-model stage count");
    }
    const std::vector<LayerRange> ranges = balanced_layer_ranges(model->n_layer(), stage_count);
    std::vector<std::shared_ptr<LlamaCppModel>> stage_models;
    std::vector<std::unique_ptr<LlamaCppStageAdapter>> stages;
    stage_models.reserve(static_cast<std::size_t>(stage_count));
    stages.reserve(static_cast<std::size_t>(stage_count));
    for (int index = 0; index < stage_count; ++index) {
        const LayerRange range = ranges[static_cast<std::size_t>(index)];
        auto stage_model = std::make_shared<LlamaCppModel>(LlamaCppModelConfig{
            .model_path = model_path,
            .n_gpu_layers = 0,
            .layer_start = range.start,
            .layer_end = range.end,
        });
        stages.push_back(std::make_unique<LlamaCppStageAdapter>(LlamaCppStageConfig{
            .model = stage_model,
            .ctx_size = 256,
            .threads = 2,
            .position = {.index = index, .count = stage_count},
            .layers = range,
        }));
        stage_models.push_back(std::move(stage_model));
    }
    verify_partition_residency(model, stage_models, ranges);

    const std::string session = "native-" + std::to_string(stage_count) + "-stage-test";
    Payload payload = text_payload(prompt);
    for (std::size_t token_index = 0; token_index < baseline_tokens.size(); ++token_index) {
        const Phase phase = token_index == 0U ? Phase::Prefill : Phase::Decode;
        for (int stage_index = 0; stage_index < stage_count; ++stage_index) {
            const std::string request_id =
                (phase == Phase::Prefill ? "prefill-" : "decode-") +
                std::to_string(token_index) + "-stage-" + std::to_string(stage_index);
            ExecutionResult result = stages[static_cast<std::size_t>(stage_index)]->execute(input_for(
                session,
                request_id,
                phase,
                static_cast<int>(token_index),
                {.index = stage_index, .count = stage_count},
                ranges[static_cast<std::size_t>(stage_index)],
                std::move(payload)
            ));
            require_success(result, request_id.c_str());
            payload = std::move(result.output.payload);
            if (stage_index < stage_count - 1 &&
                (payload.kind != PayloadKind::Activation || payload.tensor.dtype != "f32" ||
                 payload.tensor.shape.size() != 2 || payload.tensor.shape[1] != model->n_embd())) {
                throw std::runtime_error("non-final stage returned an invalid activation descriptor");
            }
        }
        if (sampled_token(payload) != baseline_tokens[token_index]) {
            throw std::runtime_error("pipeline token does not match full-model greedy baseline");
        }
    }

    for (const auto& stage : stages) {
        if (stage->session_count() != 1U) {
            throw std::runtime_error("pipeline stage session context was not retained");
        }
        stage->close_session(session);
        if (stage->session_count() != 0U) {
            throw std::runtime_error("pipeline stage session cleanup failed");
        }
    }
    if (stage_count == 2) {
        verify_idle_session_expiry(stage_models.front(), ranges.front().end);
    }
}

int run_test(const std::shared_ptr<LlamaCppModel>& model, const std::string& model_path) {
    if (model->n_layer() < 3) {
        throw std::runtime_error("partial-layer test requires at least three transformer layers");
    }
    const std::string prompt = "Once upon a time";
    LlamaCppAdapter baseline(model, 256, 2);
    const auto baseline_response = baseline.generate({.prompt = prompt, .max_tokens = integration_max_tokens});
    if (baseline_response.token_ids.size() != integration_max_tokens) {
        throw std::runtime_error("baseline must generate four tokens to validate repeated decode");
    }

    verify_single_stage_equivalence(model, prompt, baseline_response.token_ids);
    verify_pipeline_equivalence(model, model_path, prompt, baseline_response.token_ids, 2);
    verify_pipeline_equivalence(model, model_path, prompt, baseline_response.token_ids, 3);

    std::cout << "llama.cpp single-stage, two-stage, and three-stage equivalence passed: architecture="
              << model->architecture() << " layers=" << model->n_layer()
              << " tokens=" << baseline_response.token_ids.size() << '\n';
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
            const auto response = baseline.generate({.prompt = baseline_prompt_from_environment(), .max_tokens = 1});
            if (response.token_ids.empty()) {
                throw std::runtime_error("baseline did not generate a token");
            }
            std::cout << response.token_ids[0] << '\n';
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--baseline-tokens") {
            LlamaCppAdapter baseline(model, 256, 2);
            const int max_tokens = baseline_max_tokens_from_environment();
            const auto response = baseline.generate({.prompt = baseline_prompt_from_environment(), .max_tokens = max_tokens});
            print_token_ids(response.token_ids);
            return 0;
        }
        return run_test(model, model_path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
