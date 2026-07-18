#include "inference/stage.hpp"

#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::inference {
namespace {

constexpr std::uint64_t kMaxPayloadBytes = 512ULL << 20;

std::uint64_t dtype_width(const std::string& dtype) {
    if (dtype == "u8" || dtype == "i8") return 1;
    if (dtype == "f16" || dtype == "bf16") return 2;
    if (dtype == "i32" || dtype == "u32" || dtype == "f32") return 4;
    if (dtype == "i64" || dtype == "u64" || dtype == "f64") return 8;
    return 0;
}

ExecutionResult error_result(ErrorKind kind, std::string code, std::string message) {
    ExecutionResult result;
    result.error = ExecutionError{
        .kind = kind,
        .code = std::move(code),
        .message = std::move(message),
    };
    return result;
}

} // namespace

bool StagePosition::is_first() const {
    return count > 0 && index == 0;
}

bool StagePosition::is_last() const {
    return count > 0 && index >= 0 && index < count && index == count - 1;
}

bool StagePosition::is_intermediate() const {
    return count > 0 && index > 0 && index < count - 1;
}

ExecutionResult ExecutionResult::success(StageOutput output) {
    ExecutionResult result;
    result.ok = true;
    result.output = std::move(output);
    return result;
}

ExecutionResult ExecutionResult::invalid_input(std::string code, std::string message) {
    return error_result(ErrorKind::InvalidInput, std::move(code), std::move(message));
}

ExecutionResult ExecutionResult::failure(std::string code, std::string message) {
    return error_result(ErrorKind::ExecutionFailure, std::move(code), std::move(message));
}

std::string to_string(Phase phase) {
    switch (phase) {
    case Phase::Prefill:
        return "prefill";
    case Phase::Decode:
        return "decode";
    }
    throw std::invalid_argument("invalid inference phase");
}

Phase parse_phase(const std::string& value) {
    if (value == "prefill") return Phase::Prefill;
    if (value == "decode") return Phase::Decode;
    throw std::invalid_argument("invalid inference phase: " + value);
}

std::string to_string(PayloadKind kind) {
    switch (kind) {
    case PayloadKind::Text:
        return "text";
    case PayloadKind::Tokens:
        return "tokens";
    case PayloadKind::Activation:
        return "activation";
    case PayloadKind::SampledToken:
        return "sampled_token";
    }
    throw std::invalid_argument("invalid inference payload kind");
}

PayloadKind parse_payload_kind(const std::string& value) {
    if (value == "text") return PayloadKind::Text;
    if (value == "tokens") return PayloadKind::Tokens;
    if (value == "activation") return PayloadKind::Activation;
    if (value == "sampled_token") return PayloadKind::SampledToken;
    throw std::invalid_argument("invalid inference payload kind: " + value);
}

std::string validate_payload(const Payload& payload) {
    if (payload.kind == PayloadKind::Text) {
        if (payload.encoding != "utf-8") {
            return "text payload encoding must be utf-8";
        }
        if (!payload.tensor.dtype.empty() || !payload.tensor.shape.empty() ||
            !payload.tensor.byte_order.empty() || !payload.tensor.layout.empty()) {
            return "text payload must not declare tensor metadata";
        }
        return "";
    }

    if (!payload.encoding.empty()) {
        return "tensor payload must not declare text encoding";
    }
    if (payload.tensor.dtype.empty() || payload.tensor.shape.empty()) {
        return "tensor payload requires dtype and shape";
    }
    if (payload.tensor.byte_order != "little" || payload.tensor.layout != "row_major") {
        return "tensor payload requires little byte order and row_major layout";
    }

    const std::uint64_t width = dtype_width(payload.tensor.dtype);
    if (width == 0) {
        return "unsupported tensor dtype: " + payload.tensor.dtype;
    }

    std::uint64_t count = 1;
    for (const std::int64_t dimension : payload.tensor.shape) {
        if (dimension <= 0 || count > kMaxPayloadBytes / static_cast<std::uint64_t>(dimension)) {
            return "invalid or oversized tensor shape";
        }
        count *= static_cast<std::uint64_t>(dimension);
    }
    if (count > kMaxPayloadBytes / width || count * width != payload.bytes.size()) {
        return "tensor shape and dtype do not match payload length";
    }
    if (payload.kind == PayloadKind::SampledToken && count != 1) {
        return "sampled_token payload must contain exactly one element";
    }
    return "";
}

std::string validate_stage_input(const StageInput& input) {
    if (input.session_id.empty()) return "session_id is required";
    if (input.request_id.empty()) return "request_id is required";
    if (input.model_id.empty()) return "model_id is required";
    if (input.position.count <= 0) return "stage count must be greater than zero";
    if (input.position.index < 0 || input.position.index >= input.position.count) {
        return "stage index must be within stage count";
    }
    if (input.layers.end <= input.layers.start) {
        return "layer range end must be greater than start";
    }
    if (input.phase == Phase::Prefill && input.decode_step != 0) {
        return "prefill requires decode_step 0";
    }
    if (input.phase == Phase::Decode && input.decode_step <= 0) {
        return "decode requires a positive decode_step";
    }
    if (input.max_tokens <= 0) return "max_tokens must be greater than zero";
    return validate_payload(input.payload);
}

bool is_allowed_input(Phase phase, StagePosition position, PayloadKind kind) {
    if (position.count <= 0 || position.index < 0 || position.index >= position.count) {
        return false;
    }
    if (!position.is_first()) {
        return kind == PayloadKind::Activation;
    }
    if (phase == Phase::Prefill) {
        return kind == PayloadKind::Text || kind == PayloadKind::Tokens;
    }
    return kind == PayloadKind::SampledToken;
}

PayloadKind expected_output(Phase /*phase*/, StagePosition position) {
    if (position.count <= 0 || position.index < 0 || position.index >= position.count) {
        throw std::invalid_argument("invalid stage position");
    }
    return position.is_last() ? PayloadKind::SampledToken : PayloadKind::Activation;
}

} // namespace jetsonfabric::runtime::inference
