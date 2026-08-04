#include "adapters/native_stage_executor.hpp"

#include "inference/token_payload.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using jetsonfabric::native::Backend;
using jetsonfabric::native::EngineOptions;
using jetsonfabric::native::NativeEngine;
using jetsonfabric::runtime::adapters::NativeStageConfig;
using jetsonfabric::runtime::adapters::NativeStageExecutor;
using jetsonfabric::runtime::inference::ExecutionResult;
using jetsonfabric::runtime::inference::LayerRange;
using jetsonfabric::runtime::inference::Payload;
using jetsonfabric::runtime::inference::PayloadKind;
using jetsonfabric::runtime::inference::Phase;
using jetsonfabric::runtime::inference::StageInput;
using jetsonfabric::runtime::inference::StagePosition;
using jetsonfabric::runtime::tokenization::Tokenizer;

class FixtureTokenizer final : public Tokenizer {
public:
    std::vector<std::int32_t> tokenize(std::string_view text) const override {
        if (text != "fixture") throw std::invalid_argument("unexpected fixture prompt");
        return {2};
    }

    std::string token_piece(std::int32_t token) const override {
        if (token != 7) throw std::invalid_argument("unexpected fixture token");
        return "x";
    }

    bool is_end_token(std::int32_t /*token*/) const override {
        return false;
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Payload text_payload() {
    const std::string text = "fixture";
    Payload payload;
    payload.kind = PayloadKind::Text;
    payload.encoding = "utf-8";
    payload.bytes.assign(text.begin(), text.end());
    return payload;
}

StageInput prefill_input(int layer_count) {
    return StageInput{
        .session_id = "native-serving-session",
        .request_id = "native-serving-request",
        .model_id = "fixture",
        .phase = Phase::Prefill,
        .decode_step = 0,
        .position = StagePosition{.index = 0, .count = 1},
        .layers = LayerRange{.start = 0, .end = layer_count},
        .payload = text_payload(),
        .max_tokens = 2,
    };
}

void require_sampled_token(const ExecutionResult& result) {
    require(result.ok, "native stage execution failed: " + result.error.message);
    require(
        jetsonfabric::runtime::inference::decode_single_token(result.output.payload) == 7,
        "native stage sampled an unexpected token"
    );
    require(result.output.token_text == "x", "native stage returned unexpected text");
    require(result.output.completion_tokens == 1, "native stage did not count its token");
}

void run_serving_test(const std::string& package_path) {
    auto engine = std::make_shared<NativeEngine>(package_path, Backend::Cpu, 1);
    const int layer_count = static_cast<int>(engine->model_info().layer_count);
    const auto one_session = jetsonfabric::native::estimate_stage_memory(
        package_path,
        Backend::Cpu,
        0,
        static_cast<std::uint32_t>(layer_count),
        16,
        1
    );
    const auto two_sessions = jetsonfabric::native::estimate_stage_memory(
        package_path,
        Backend::Cpu,
        0,
        static_cast<std::uint32_t>(layer_count),
        16,
        2
    );
    require(
        one_session.resident_weight_bytes == engine->model_info().weight_bytes,
        "native stage estimate did not match resident weights"
    );
    require(one_session.reserved_kv_bytes > 0, "native stage estimate omitted KV memory");
    require(
        two_sessions.reserved_kv_bytes == 2 * one_session.reserved_kv_bytes,
        "native stage estimate did not scale with parallel sessions"
    );

    std::unique_ptr<jetsonfabric::native::NativeSession> direct_session =
        engine->create_session(2);
    const std::array<std::int32_t, 1> direct_prompt{2};
    require(direct_session->position() == 0, "native session did not start empty");
    require(direct_session->prefill_greedy(direct_prompt) == 7,
            "native session prefill token mismatch");
    require(direct_session->position() == 1, "native prefill position mismatch");
    require(direct_session->decode_greedy(7) == 7, "native session decode mismatch");
    require(direct_session->position() == 2, "native decode position mismatch");
    direct_session->rollback(1);
    require(direct_session->position() == 1, "native rollback position mismatch");

    NativeStageExecutor executor(NativeStageConfig{
        .engine = engine,
        .tokenizer = std::make_shared<FixtureTokenizer>(),
        .ctx_size = 16,
        .max_parallel_sessions = 1,
        .position = StagePosition{.index = 0, .count = 1},
        .layers = LayerRange{.start = 0, .end = layer_count},
    });

    const StageInput prefill = prefill_input(layer_count);
    const ExecutionResult prefill_result = executor.execute(prefill);
    require_sampled_token(prefill_result);
    require(prefill_result.output.prompt_tokens == 1, "native prefill token count mismatch");
    require(
        prefill_result.output.prompt_token_ids == std::vector<std::uint32_t>{2},
        "native prefill token IDs mismatch"
    );
    require(executor.session_count() == 1, "native prefill did not retain a session");

    const ExecutionResult duplicate = executor.execute(prefill);
    require(!duplicate.ok, "native stage accepted a duplicate prefill session");
    require(duplicate.error.code == "duplicate_stage_session", "unexpected duplicate error");

    StageInput capacity_prefill = prefill;
    capacity_prefill.session_id = "native-serving-session-2";
    const ExecutionResult at_capacity = executor.execute(capacity_prefill);
    require(!at_capacity.ok, "native stage exceeded its configured session capacity");
    require(
        at_capacity.error.code == "stage_session_capacity_exceeded",
        "native stage capacity rejection used the wrong error"
    );

    StageInput decode = prefill;
    decode.phase = Phase::Decode;
    decode.decode_step = 1;
    decode.payload = jetsonfabric::runtime::inference::sampled_token_payload(7);
    require_sampled_token(executor.execute(decode));

    executor.rollback_session(prefill.session_id, 1);
    require_sampled_token(executor.execute(decode));

    executor.close_session(prefill.session_id);
    require(executor.session_count() == 0, "native close did not release the session");
    require_sampled_token(executor.execute(capacity_prefill));
    executor.close_session(capacity_prefill.session_id);
    const ExecutionResult after_close = executor.execute(decode);
    require(!after_close.ok, "native stage decoded after session close");
    require(after_close.error.code == "stage_session_not_found", "unexpected close error");
}

void run_distributed_serving_test(const std::string& package_path) {
    auto first_engine = std::make_shared<NativeEngine>(
        package_path,
        EngineOptions{
            .backend = Backend::Cpu,
            .threads = 1,
            .layer_start = 0,
            .layer_end = 1,
        }
    );
    auto final_engine = std::make_shared<NativeEngine>(
        package_path,
        EngineOptions{
            .backend = Backend::Cpu,
            .threads = 1,
            .layer_start = 1,
            .layer_end = 2,
        }
    );
    require(
        first_engine->model_info().resident_layer_start == 0 &&
            first_engine->model_info().resident_layer_end == 1,
        "first native stage loaded the wrong layer range"
    );
    require(
        final_engine->model_info().resident_layer_start == 1 &&
            final_engine->model_info().resident_layer_end == 2,
        "final native stage loaded the wrong layer range"
    );
    require(
        first_engine->model_info().weight_bytes <
            first_engine->model_info().total_weight_bytes &&
            final_engine->model_info().weight_bytes <
                final_engine->model_info().total_weight_bytes,
        "native stages did not reduce resident model weights"
    );

    auto tokenizer = std::make_shared<FixtureTokenizer>();
    NativeStageExecutor first(NativeStageConfig{
        .engine = first_engine,
        .tokenizer = tokenizer,
        .ctx_size = 16,
        .max_parallel_sessions = 1,
        .position = StagePosition{.index = 0, .count = 2},
        .layers = LayerRange{.start = 0, .end = 1},
    });
    NativeStageExecutor final(NativeStageConfig{
        .engine = final_engine,
        .tokenizer = tokenizer,
        .ctx_size = 16,
        .max_parallel_sessions = 1,
        .position = StagePosition{.index = 1, .count = 2},
        .layers = LayerRange{.start = 1, .end = 2},
    });

    StageInput first_prefill = prefill_input(1);
    first_prefill.position = StagePosition{.index = 0, .count = 2};
    const ExecutionResult first_prefill_result = first.execute(first_prefill);
    require(first_prefill_result.ok, "first native prefill stage failed");
    require(
        first_prefill_result.output.payload.kind == PayloadKind::Activation,
        "first native stage did not return an activation"
    );

    StageInput final_prefill = first_prefill;
    final_prefill.position = StagePosition{.index = 1, .count = 2};
    final_prefill.layers = LayerRange{.start = 1, .end = 2};
    final_prefill.payload = first_prefill_result.output.payload;
    require_sampled_token(final.execute(final_prefill));

    StageInput oversized_activation = final_prefill;
    oversized_activation.session_id = "oversized-activation";
    oversized_activation.payload.tensor.shape[0] =
        std::numeric_limits<std::int64_t>::max();
    oversized_activation.payload.bytes.clear();
    const ExecutionResult oversized_result = final.execute(oversized_activation);
    require(!oversized_result.ok, "native stage accepted an oversized activation shape");
    require(
        oversized_result.error.code == "invalid_native_stage_input",
        "oversized activation returned the wrong error"
    );

    StageInput first_decode = first_prefill;
    first_decode.phase = Phase::Decode;
    first_decode.decode_step = 1;
    first_decode.payload = jetsonfabric::runtime::inference::sampled_token_payload(7);
    const ExecutionResult first_decode_result = first.execute(first_decode);
    require(first_decode_result.ok, "first native decode stage failed");
    require(
        first_decode_result.output.payload.kind == PayloadKind::Activation,
        "first native decode stage did not return an activation"
    );

    StageInput final_decode = final_prefill;
    final_decode.phase = Phase::Decode;
    final_decode.decode_step = 1;
    final_decode.payload = first_decode_result.output.payload;
    require_sampled_token(final.execute(final_decode));
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("JFM package path is required");
        run_serving_test(argv[1]);
        run_distributed_serving_test(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native serving test: " << error.what() << '\n';
        return 1;
    }
}
