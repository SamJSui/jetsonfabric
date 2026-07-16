#include "pipeline_parallel/synthetic_activation_executor.hpp"

#include <bit>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

StageRunResult error_result(const std::string& code, const std::string& message) {
    return StageRunResult{false, "400 Bad Request", code, message, {}};
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

std::vector<std::uint8_t> make_activation(const std::vector<std::uint8_t>& input) {
    const std::uint32_t seed = protocol::payload_crc32(input);
    std::vector<std::uint8_t> activation;
    activation.reserve(4U * 16U * sizeof(float));
    for (std::uint32_t index = 0; index < 64U; ++index) {
        const std::uint32_t mixed = seed ^ (0x9e3779b9U * (index + 1U));
        const float value = static_cast<float>(mixed & 0xffffU) / 65535.0F;
        append_u32_le(activation, std::bit_cast<std::uint32_t>(value));
    }
    return activation;
}

protocol::StageResponse base_response(const protocol::StageRequest& request) {
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
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    return response;
}

StageRunResult activation_result(const protocol::StageRequest& request, std::vector<std::uint8_t> activation) {
    protocol::StageResponse response = base_response(request);
    response.payload_kind = "activation";
    response.dtype = "f32";
    response.shape = {4, 16};
    response.byte_order = "little";
    response.layout = "row_major";
    response.payload = std::move(activation);
    response.bytes_out = static_cast<std::int64_t>(response.payload.size());
    return StageRunResult{true, "200 OK", "", "", std::move(response)};
}

StageRunResult sampled_token_result(const protocol::StageRequest& request, const std::vector<std::uint8_t>& activation) {
    protocol::StageResponse response = base_response(request);
    response.payload_kind = "sampled_token";
    response.dtype = "u32";
    response.shape = {1};
    response.byte_order = "little";
    response.layout = "row_major";
    append_u32_le(response.payload, protocol::payload_crc32(activation));
    response.bytes_out = static_cast<std::int64_t>(response.payload.size());
    response.completion_tokens = 1;
    return StageRunResult{true, "200 OK", "", "", std::move(response)};
}

} // namespace

StageRunResult SyntheticActivationExecutor::run_layers(const protocol::StageRequest& request) const {
    if (request.is_first_stage()) {
        const bool valid_input = request.phase == "prefill"
            ? request.payload_kind == "text" || request.payload_kind == "tokens"
            : request.payload_kind == "sampled_token";
        if (!valid_input) {
            return error_result("invalid_synthetic_input", "first synthetic stage received an invalid payload kind");
        }
        std::vector<std::uint8_t> activation = make_activation(request.payload);
        if (request.is_last_stage()) {
            return sampled_token_result(request, activation);
        }
        return activation_result(request, std::move(activation));
    }

    if (request.payload_kind != "activation" || request.dtype != "f32" ||
        request.shape != std::vector<std::int64_t>({4, 16}) || request.payload.size() != 256U) {
        return error_result("invalid_synthetic_activation", "synthetic downstream stage requires f32[4,16] activation bytes");
    }
    if (request.is_last_stage()) {
        return sampled_token_result(request, request.payload);
    }
    return activation_result(request, request.payload);
}

} // namespace jetsonfabric::runtime::pipeline_parallel
