#include "adapters/native_stage_executor.hpp"

#include "inference/token_payload.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using jetsonfabric::native::Backend;
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

    StageInput decode = prefill;
    decode.phase = Phase::Decode;
    decode.decode_step = 1;
    decode.payload = jetsonfabric::runtime::inference::sampled_token_payload(7);
    require_sampled_token(executor.execute(decode));

    executor.rollback_session(prefill.session_id, 1);
    require_sampled_token(executor.execute(decode));

    executor.close_session(prefill.session_id);
    require(executor.session_count() == 0, "native close did not release the session");
    const ExecutionResult after_close = executor.execute(decode);
    require(!after_close.ok, "native stage decoded after session close");
    require(after_close.error.code == "stage_session_not_found", "unexpected close error");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("JFM package path is required");
        run_serving_test(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native serving test: " << error.what() << '\n';
        return 1;
    }
}
