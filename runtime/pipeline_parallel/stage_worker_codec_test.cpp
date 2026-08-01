#include "activation/activation_codec_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace runtime = jetsonfabric::runtime;

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

std::vector<std::uint8_t> f32_bytes(float first, float second) {
    std::vector<std::uint8_t> bytes;
    for (const float value : {first, second}) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        bytes.push_back(static_cast<std::uint8_t>(bits & 0xffU));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xffU));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xffU));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 24U) & 0xffU));
    }
    return bytes;
}

class FirstStageExecutor final : public runtime::pipeline_parallel::LayerExecutor {
public:
    runtime::inference::ExecutionResult execute(
        const runtime::inference::StageInput& input
    ) const override {
        expect(
            input.payload.kind == runtime::inference::PayloadKind::Text,
            "first stage input changed before engine execution"
        );
        return runtime::inference::ExecutionResult::success(
            runtime::inference::StageOutput{
                .payload = runtime::inference::Payload{
                    .kind = runtime::inference::PayloadKind::Activation,
                    .encoding = "",
                    .tensor = runtime::inference::TensorDescriptor{
                        .dtype = "f32",
                        .shape = {1, 2},
                        .byte_order = "little",
                        .layout = "row_major",
                    },
                    .bytes = f32_bytes(1.0F, -2.0F),
                },
                .prompt_tokens = 0,
                .completion_tokens = 0,
                .token_text = "",
                .end_of_generation = false,
            }
        );
    }
};

class LastStageExecutor final : public runtime::pipeline_parallel::LayerExecutor {
public:
    runtime::inference::ExecutionResult execute(
        const runtime::inference::StageInput& input
    ) const override {
        expect(input.payload.tensor.dtype == "f32", "engine did not receive decoded f32");
        expect(input.payload.bytes == f32_bytes(1.0F, -2.0F), "decoded activation changed");
        return runtime::inference::ExecutionResult::success(
            runtime::inference::StageOutput{
                .payload = runtime::inference::Payload{
                    .kind = runtime::inference::PayloadKind::SampledToken,
                    .encoding = "",
                    .tensor = runtime::inference::TensorDescriptor{
                        .dtype = "u32",
                        .shape = {1},
                        .byte_order = "little",
                        .layout = "row_major",
                    },
                    .bytes = {42, 0, 0, 0},
                },
                .prompt_tokens = 0,
                .completion_tokens = 1,
                .token_text = "",
                .end_of_generation = false,
            }
        );
    }
};

runtime::protocol::StageRequest request_for(
    int stage_index,
    const std::string& node_name
) {
    return runtime::protocol::StageRequest{
        .session_id = "session-1",
        .request_id = "request-1",
        .model_id = "model-a",
        .deployment_id = "",
        .deployment_epoch = 0,
        .model_sha256 = "",
        .phase = "prefill",
        .decode_step = 0,
        .stage_index = stage_index,
        .stage_count = 2,
        .node_name = node_name,
        .layer_start = stage_index == 0 ? 0 : 2,
        .layer_end = stage_index == 0 ? 2 : 4,
        .payload_kind = "text",
        .encoding = "utf-8",
        .dtype = "",
        .shape = {},
        .byte_order = "",
        .layout = "",
        .payload = {'h', 'i'},
        .max_tokens = 1,
    };
}

void test_stage_boundary_uses_f16_on_wire_and_f32_in_engine() {
    const auto codec =
        runtime::activation::make_default_activation_codec_factory()->create_codec("f16");
    const FirstStageExecutor first_executor;
    const LastStageExecutor last_executor;
    const runtime::pipeline_parallel::StageWorker first(
        "node-a",
        "model-a",
        runtime::pipeline_parallel::StageAssignment{
            .stage_index = 0,
            .stage_count = 2,
            .layer_start = 0,
            .layer_end = 2,
        },
        first_executor,
        codec
    );
    const runtime::pipeline_parallel::StageWorker last(
        "node-b",
        "model-a",
        runtime::pipeline_parallel::StageAssignment{
            .stage_index = 1,
            .stage_count = 2,
            .layer_start = 2,
            .layer_end = 4,
        },
        last_executor,
        codec
    );

    const auto first_result = first.run(request_for(0, "node-a"));
    expect(first_result.ok, "first stage failed");
    expect(first_result.response.dtype == "f16", "first stage did not emit f16");
    expect(first_result.response.payload.size() == 4, "f16 wire payload has wrong size");

    runtime::protocol::StageRequest last_request = request_for(1, "node-b");
    last_request.payload_kind = first_result.response.payload_kind;
    last_request.encoding = first_result.response.encoding;
    last_request.dtype = first_result.response.dtype;
    last_request.shape = first_result.response.shape;
    last_request.byte_order = first_result.response.byte_order;
    last_request.layout = first_result.response.layout;
    last_request.payload = first_result.response.payload;

    const auto last_result = last.run(last_request);
    expect(last_result.ok, "last stage failed");
    expect(last_result.response.payload_kind == "sampled_token", "last stage output changed");
}

} // namespace

int main() {
    test_stage_boundary_uses_f16_on_wire_and_f32_in_engine();
    std::cout << "stage worker codec tests passed\n";
    return 0;
}
