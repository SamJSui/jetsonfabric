#include "inference/token_payload.hpp"

#include <stdexcept>

namespace jetsonfabric::runtime::inference {
namespace {

void append_i32_le(std::vector<std::uint8_t>& output, std::int32_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    output.push_back(static_cast<std::uint8_t>(bits & 0xffU));
    output.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xffU));
    output.push_back(static_cast<std::uint8_t>((bits >> 24U) & 0xffU));
}

std::int32_t read_i32_le(const std::uint8_t* data) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(data[0]) |
        (static_cast<std::uint32_t>(data[1]) << 8U) |
        (static_cast<std::uint32_t>(data[2]) << 16U) |
        (static_cast<std::uint32_t>(data[3]) << 24U);
    return static_cast<std::int32_t>(bits);
}

TensorDescriptor token_descriptor(std::size_t count) {
    return TensorDescriptor{
        .dtype = "i32",
        .shape = {static_cast<std::int64_t>(count)},
        .byte_order = "little",
        .layout = "row_major",
    };
}

} // namespace

std::vector<std::int32_t> decode_token_ids(const Payload& payload) {
    if ((payload.tensor.dtype != "i32" && payload.tensor.dtype != "u32") ||
        payload.tensor.shape.size() != 1 || payload.tensor.shape[0] <= 0 ||
        payload.tensor.byte_order != "little" ||
        payload.tensor.layout != "row_major" ||
        payload.bytes.size() % sizeof(std::int32_t) != 0U ||
        static_cast<std::uint64_t>(payload.tensor.shape[0]) !=
            payload.bytes.size() / sizeof(std::int32_t)) {
        throw std::invalid_argument(
            "token payload must be a little-endian i32 or u32 vector"
        );
    }
    std::vector<std::int32_t> tokens(payload.bytes.size() / sizeof(std::int32_t));
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = read_i32_le(payload.bytes.data() + index * sizeof(std::int32_t));
    }
    return tokens;
}

std::int32_t decode_single_token(const Payload& payload) {
    const std::vector<std::int32_t> tokens = decode_token_ids(payload);
    if (tokens.size() != 1U) {
        throw std::invalid_argument("token payload must contain exactly one token");
    }
    return tokens.front();
}

Payload sampled_token_payload(std::int32_t token) {
    Payload payload;
    payload.kind = PayloadKind::SampledToken;
    payload.tensor = token_descriptor(1);
    append_i32_le(payload.bytes, token);
    return payload;
}

Payload sampled_tokens_payload(std::span<const std::int32_t> tokens) {
    if (tokens.size() < 2U) {
        throw std::invalid_argument("sampled_tokens payload requires at least two tokens");
    }
    Payload payload;
    payload.kind = PayloadKind::SampledTokens;
    payload.tensor = token_descriptor(tokens.size());
    payload.bytes.reserve(tokens.size() * sizeof(std::int32_t));
    for (const std::int32_t token : tokens) append_i32_le(payload.bytes, token);
    return payload;
}

} // namespace jetsonfabric::runtime::inference
