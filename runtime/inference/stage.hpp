#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::inference {

enum class Phase {
    Prefill,
    Decode,
};

enum class PayloadKind {
    Text,
    Tokens,
    Activation,
    SampledToken,
    SampledTokens,
};

enum class ErrorKind {
    InvalidInput,
    ExecutionFailure,
};

struct StagePosition {
    int index = 0;
    int count = 1;

    bool is_first() const;
    bool is_last() const;
};

struct LayerRange {
    int start = 0;
    int end = 0;
};

struct TensorDescriptor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string byte_order;
    std::string layout;
};

struct Payload {
    PayloadKind kind = PayloadKind::Text;
    std::string encoding;
    TensorDescriptor tensor;
    std::vector<std::uint8_t> bytes;
};

struct StageInput {
    std::string session_id;
    std::string request_id;
    std::string model_id;

    Phase phase = Phase::Prefill;
    int decode_step = 0;

    StagePosition position;
    LayerRange layers;
    Payload payload;

    int max_tokens = 128;
};

struct StageOutput {
    Payload payload;
    int prompt_tokens = 0;
    std::vector<std::uint32_t> prompt_token_ids;
    int completion_tokens = 0;
    int execution_batch_size = 1;
    int verification_width = 1;
    std::string token_text;
    std::vector<std::uint32_t> token_text_offsets;
    std::vector<std::uint8_t> token_eog;
    bool end_of_generation = false;
};

struct ExecutionError {
    ErrorKind kind = ErrorKind::ExecutionFailure;
    std::string code;
    std::string message;
};

struct ExecutionResult {
    bool ok = false;
    StageOutput output;
    ExecutionError error;

    static ExecutionResult success(StageOutput output);
    static ExecutionResult invalid_input(std::string code, std::string message);
    static ExecutionResult failure(std::string code, std::string message);
};

std::string to_string(Phase phase);
Phase parse_phase(const std::string& value);

std::string to_string(PayloadKind kind);
PayloadKind parse_payload_kind(const std::string& value);

std::string validate_payload(const Payload& payload);
std::string validate_stage_input(const StageInput& input);

bool is_allowed_input(Phase phase, StagePosition position, PayloadKind kind);
PayloadKind expected_output(Phase phase, StagePosition position);

} // namespace jetsonfabric::runtime::inference
