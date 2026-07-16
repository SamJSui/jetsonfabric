#include "inference/stage.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace inference = jetsonfabric::runtime::inference;

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

inference::Payload activation_payload() {
    return inference::Payload{
        .kind = inference::PayloadKind::Activation,
        .encoding = "",
        .tensor = inference::TensorDescriptor{
            .dtype = "f32",
            .shape = {4, 16},
            .byte_order = "little",
            .layout = "row_major",
        },
        .bytes = std::vector<std::uint8_t>(256, 0x5a),
    };
}

} // namespace

int main() {
    expect(inference::parse_phase("prefill") == inference::Phase::Prefill, "prefill parsing failed");
    expect(inference::parse_phase("decode") == inference::Phase::Decode, "decode parsing failed");
    expect(inference::parse_payload_kind("activation") == inference::PayloadKind::Activation,
           "activation parsing failed");

    inference::StageInput input{
        .session_id = "session-1",
        .request_id = "request-1",
        .model_id = "model-1",
        .phase = inference::Phase::Prefill,
        .decode_step = 0,
        .position = inference::StagePosition{.index = 1, .count = 2},
        .layers = inference::LayerRange{.start = 14, .end = 28},
        .payload = activation_payload(),
        .max_tokens = 8,
    };
    expect(inference::validate_stage_input(input).empty(), "valid typed stage input was rejected");
    expect(inference::is_allowed_input(input.phase, input.position, input.payload.kind),
           "valid downstream activation was rejected");
    expect(inference::expected_output(input.phase, input.position) == inference::PayloadKind::SampledToken,
           "final stage output kind was incorrect");

    input.payload.bytes.pop_back();
    expect(!inference::validate_stage_input(input).empty(), "truncated activation was accepted");

    inference::StagePosition first{.index = 0, .count = 2};
    expect(inference::is_allowed_input(inference::Phase::Prefill, first, inference::PayloadKind::Text),
           "prefill text input was rejected");
    expect(inference::is_allowed_input(inference::Phase::Decode, first, inference::PayloadKind::SampledToken),
           "decode sampled token input was rejected");
    expect(!inference::is_allowed_input(inference::Phase::Decode, first, inference::PayloadKind::Text),
           "decode text input was accepted");

    inference::StageOutput output{.payload = activation_payload()};
    const inference::ExecutionResult result = inference::ExecutionResult::success(std::move(output));
    expect(result.ok && result.output.payload.kind == inference::PayloadKind::Activation,
           "successful execution result lost typed payload");

    std::cout << "runtime stage interface tests passed\n";
    return 0;
}
