#include "pipeline_parallel/synthetic_activation_executor.hpp"

#include <bit>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

std::uint32_t crc32(const std::vector<std::uint8_t>& payload) {
    std::uint32_t crc = 0xffffffffU;
    for (const std::uint8_t value : payload) {
        crc ^= value;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

std::vector<std::uint8_t> make_activation(const std::vector<std::uint8_t>& input) {
    const std::uint32_t seed = crc32(input);
    std::vector<std::uint8_t> activation;
    activation.reserve(4U * 16U * sizeof(float));
    for (std::uint32_t index = 0; index < 64U; ++index) {
        const std::uint32_t mixed = seed ^ (0x9e3779b9U * (index + 1U));
        const float value = static_cast<float>(mixed & 0xffffU) / 65535.0F;
        append_u32_le(activation, std::bit_cast<std::uint32_t>(value));
    }
    return activation;
}

inference::Payload activation_payload(std::vector<std::uint8_t> activation) {
    return inference::Payload{
        .kind = inference::PayloadKind::Activation,
        .encoding = "",
        .tensor = inference::TensorDescriptor{
            .dtype = "f32",
            .shape = {4, 16},
            .byte_order = "little",
            .layout = "row_major",
        },
        .bytes = std::move(activation),
    };
}

inference::Payload sampled_token_payload(const std::vector<std::uint8_t>& activation) {
    inference::Payload payload{
        .kind = inference::PayloadKind::SampledToken,
        .encoding = "",
        .tensor = inference::TensorDescriptor{
            .dtype = "u32",
            .shape = {1},
            .byte_order = "little",
            .layout = "row_major",
        },
        .bytes = {},
    };
    append_u32_le(payload.bytes, crc32(activation));
    return payload;
}

inference::ExecutionResult activation_result(std::vector<std::uint8_t> activation) {
    return inference::ExecutionResult::success(inference::StageOutput{
        .payload = activation_payload(std::move(activation)),
    });
}

inference::ExecutionResult sampled_token_result(const std::vector<std::uint8_t>& activation) {
    return inference::ExecutionResult::success(inference::StageOutput{
        .payload = sampled_token_payload(activation),
        .completion_tokens = 1,
    });
}

bool is_expected_activation(const inference::Payload& payload) {
    return payload.kind == inference::PayloadKind::Activation &&
           payload.tensor.dtype == "f32" &&
           payload.tensor.shape == std::vector<std::int64_t>({4, 16}) &&
           payload.tensor.byte_order == "little" &&
           payload.tensor.layout == "row_major" &&
           payload.bytes.size() == 256U;
}

} // namespace

inference::ExecutionResult SyntheticActivationExecutor::execute(
    const inference::StageInput& input
) const {
    if (!inference::is_allowed_input(input.phase, input.position, input.payload.kind)) {
        return inference::ExecutionResult::invalid_input(
            "invalid_synthetic_input",
            "synthetic stage received a payload kind that is invalid for its phase and position"
        );
    }

    if (input.position.is_first()) {
        std::vector<std::uint8_t> activation = make_activation(input.payload.bytes);
        if (input.position.is_last()) {
            return sampled_token_result(activation);
        }
        return activation_result(std::move(activation));
    }

    if (!is_expected_activation(input.payload)) {
        return inference::ExecutionResult::invalid_input(
            "invalid_synthetic_activation",
            "synthetic downstream stage requires f32[4,16] activation bytes"
        );
    }
    if (input.position.is_last()) {
        return sampled_token_result(input.payload.bytes);
    }
    return activation_result(input.payload.bytes);
}

} // namespace jetsonfabric::runtime::pipeline_parallel
