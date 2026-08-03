#include "adapters/native_stage_executor.hpp"

#include "inference/token_payload.hpp"

#include <limits>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

std::vector<float> activation_values(
    const inference::Payload& payload,
    std::uint32_t embedding_length
) {
    if (payload.kind != inference::PayloadKind::Activation ||
        payload.tensor.dtype != "f32" || payload.tensor.shape.size() != 2 ||
        payload.tensor.shape[0] <= 0 ||
        payload.tensor.shape[1] != static_cast<std::int64_t>(embedding_length)) {
        throw std::invalid_argument(
            "native activation must be f32[sequence_length, hidden_size]"
        );
    }
    const std::size_t sequence_length =
        static_cast<std::size_t>(payload.tensor.shape[0]);
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (embedding_length == 0 || sequence_length > max_size / embedding_length) {
        throw std::invalid_argument("native activation shape exceeds addressable memory");
    }
    const std::size_t value_count = sequence_length * embedding_length;
    if (value_count > max_size / sizeof(float)) {
        throw std::invalid_argument("native activation byte count exceeds addressable memory");
    }
    if (payload.bytes.size() != value_count * sizeof(float)) {
        throw std::invalid_argument("native activation byte count does not match its shape");
    }
    std::vector<float> values(value_count);
    std::memcpy(values.data(), payload.bytes.data(), payload.bytes.size());
    return values;
}

inference::Payload activation_payload(
    std::vector<float> values,
    std::uint32_t embedding_length
) {
    if (values.empty() || values.size() % embedding_length != 0) {
        throw std::logic_error("native engine returned an invalid activation shape");
    }
    inference::Payload payload;
    payload.kind = inference::PayloadKind::Activation;
    payload.tensor = inference::TensorDescriptor{
        .dtype = "f32",
        .shape = {
            static_cast<std::int64_t>(values.size() / embedding_length),
            static_cast<std::int64_t>(embedding_length),
        },
        .byte_order = "little",
        .layout = "row_major",
    };
    payload.bytes.resize(values.size() * sizeof(float));
    std::memcpy(payload.bytes.data(), values.data(), payload.bytes.size());
    return payload;
}

} // namespace

class NativeStageExecutor::Impl {
    struct SessionState {
        std::unique_ptr<native::NativeSession> session;
        int expected_decode_step = 1;
    };

public:
    explicit Impl(NativeStageConfig config_in) : config(std::move(config_in)) {
        if (!config.engine || !config.tokenizer) {
            throw std::invalid_argument("native executor requires engine and tokenizer");
        }
        if (config.ctx_size <= 0) {
            throw std::invalid_argument("native executor context size must be positive");
        }
        const int layer_count = static_cast<int>(config.engine->model_info().layer_count);
        if (config.position.count <= 0 || config.position.index < 0 ||
            config.position.index >= config.position.count ||
            config.layers.start < 0 || config.layers.end <= config.layers.start ||
            config.layers.end > layer_count ||
            (config.position.is_first() && config.layers.start != 0) ||
            (config.position.is_last() && config.layers.end != layer_count)) {
            throw std::invalid_argument("native serving stage assignment is invalid");
        }
        const native::ModelInfo& info = config.engine->model_info();
        if (config.layers.start != static_cast<int>(info.resident_layer_start) ||
            config.layers.end != static_cast<int>(info.resident_layer_end)) {
            throw std::invalid_argument("native stage exceeds its resident layer range");
        }
        if (static_cast<std::uint32_t>(config.ctx_size) >
            config.engine->model_info().context_length) {
            throw std::invalid_argument("native context size exceeds model context");
        }
    }

    inference::ExecutionResult execute(const inference::StageInput& input) {
        const std::lock_guard lock(mutex);
        try {
            const std::string input_error = inference::validate_stage_input(input);
            if (!input_error.empty()) {
                return inference::ExecutionResult::invalid_input(
                    "invalid_native_stage_input",
                    input_error
                );
            }
            if (!assignment_matches(input)) {
                return inference::ExecutionResult::invalid_input(
                    "native_stage_assignment_mismatch",
                    "stage input does not match the configured native assignment"
                );
            }
            return input.phase == inference::Phase::Prefill
                ? prefill(input)
                : decode(input);
        } catch (const std::invalid_argument& error) {
            return inference::ExecutionResult::invalid_input(
                "invalid_native_stage_input",
                error.what()
            );
        } catch (const std::exception& error) {
            return inference::ExecutionResult::failure(
                "native_stage_execution_failed",
                error.what()
            );
        }
    }

    void close_session(const std::string& session_id) {
        const std::lock_guard lock(mutex);
        sessions.erase(session_id);
    }

    void rollback_session(const std::string& session_id, int token_count) {
        const std::lock_guard lock(mutex);
        if (token_count <= 0) {
            throw std::invalid_argument("rollback token count must be positive");
        }
        const auto found = sessions.find(session_id);
        if (found == sessions.end()) {
            throw std::invalid_argument("native stage session not found");
        }
        SessionState& state = found->second;
        if (token_count >= state.expected_decode_step) {
            throw std::invalid_argument("rollback exceeds the decoded native suffix");
        }
        state.session->rollback(static_cast<std::size_t>(token_count));
        state.expected_decode_step -= token_count;
    }

    std::size_t session_count() const {
        const std::lock_guard lock(mutex);
        return sessions.size();
    }

private:
    bool assignment_matches(const inference::StageInput& input) const noexcept {
        return input.position.index == config.position.index &&
            input.position.count == config.position.count &&
            input.layers.start == config.layers.start &&
            input.layers.end == config.layers.end;
    }

    std::vector<std::int32_t> input_tokens(const inference::StageInput& input) const {
        if (input.payload.kind == inference::PayloadKind::Text) {
            return config.tokenizer->tokenize(std::string_view(
                reinterpret_cast<const char*>(input.payload.bytes.data()),
                input.payload.bytes.size()
            ));
        }
        if (input.payload.kind == inference::PayloadKind::Tokens ||
            input.payload.kind == inference::PayloadKind::SampledToken) {
            return inference::decode_token_ids(input.payload);
        }
        throw std::invalid_argument("native first stage requires text or token input");
    }

    std::size_t session_capacity(std::size_t prompt_tokens, int max_tokens) const {
        const std::size_t decode_inputs = static_cast<std::size_t>(max_tokens - 1);
        if (prompt_tokens > std::numeric_limits<std::size_t>::max() - decode_inputs) {
            throw std::invalid_argument("native session capacity overflow");
        }
        const std::size_t required = prompt_tokens + decode_inputs;
        if (required > static_cast<std::size_t>(config.ctx_size)) {
            throw std::invalid_argument("native request exceeds configured context size");
        }
        return required;
    }

    inference::StageOutput token_output(std::int32_t token) const {
        inference::StageOutput output;
        output.payload = inference::sampled_token_payload(token);
        output.end_of_generation = config.tokenizer->is_end_token(token);
        if (!output.end_of_generation) {
            output.token_text = config.tokenizer->token_piece(token);
            output.completion_tokens = 1;
        }
        return output;
    }

    inference::StageOutput stage_output(native::StageResult result) const {
        if (result.has_sampled_token()) return token_output(result.sampled_token);
        inference::StageOutput output;
        output.payload = activation_payload(
            std::move(result.activations),
            config.engine->model_info().embedding_length
        );
        output.execution_batch_size = 1;
        return output;
    }

    inference::ExecutionResult prefill(const inference::StageInput& input) {
        if (sessions.contains(input.session_id)) {
            return inference::ExecutionResult::invalid_input(
                "duplicate_stage_session",
                "prefill session already exists on this native stage"
            );
        }
        std::vector<std::int32_t> tokens;
        std::vector<float> activations;
        std::size_t token_count = 0;
        if (config.position.is_first()) {
            tokens = input_tokens(input);
            token_count = tokens.size();
        } else {
            activations = activation_values(
                input.payload,
                config.engine->model_info().embedding_length
            );
            token_count = activations.size() /
                config.engine->model_info().embedding_length;
        }
        const std::size_t capacity = session_capacity(token_count, input.max_tokens);
        std::unique_ptr<native::NativeSession> session =
            config.engine->create_session(capacity);
        native::StageResult stage_result = config.position.is_first()
            ? session->prefill_stage_tokens(tokens)
            : session->prefill_stage_activations(activations, token_count);
        inference::StageOutput output = stage_output(std::move(stage_result));
        if (config.position.is_first()) {
            output.prompt_tokens = static_cast<int>(tokens.size());
            output.prompt_token_ids.reserve(tokens.size());
            for (const std::int32_t token : tokens) {
                if (token < 0) {
                    throw std::invalid_argument("native tokenizer returned a negative token");
                }
                output.prompt_token_ids.push_back(static_cast<std::uint32_t>(token));
            }
        }
        sessions.emplace(
            input.session_id,
            SessionState{.session = std::move(session), .expected_decode_step = 1}
        );
        return inference::ExecutionResult::success(std::move(output));
    }

    inference::ExecutionResult decode(const inference::StageInput& input) {
        const auto found = sessions.find(input.session_id);
        if (found == sessions.end()) {
            return inference::ExecutionResult::invalid_input(
                "stage_session_not_found",
                "decode requires an existing native prefill session"
            );
        }
        SessionState& state = found->second;
        if (input.decode_step != state.expected_decode_step) {
            return inference::ExecutionResult::invalid_input(
                "unexpected_decode_step",
                "native decode step does not match session state"
            );
        }
        native::StageResult stage_result;
        if (config.position.is_first()) {
            const std::vector<std::int32_t> tokens = input_tokens(input);
            if (tokens.size() != 1U) {
                return inference::ExecutionResult::invalid_input(
                    "native_multi_token_verification_unsupported",
                    "native serving currently accepts one decode token per pass"
                );
            }
            stage_result = state.session->decode_stage_token(tokens.front());
        } else {
            std::vector<float> activation = activation_values(
                input.payload,
                config.engine->model_info().embedding_length
            );
            if (activation.size() != config.engine->model_info().embedding_length) {
                return inference::ExecutionResult::invalid_input(
                    "native_multi_token_verification_unsupported",
                    "native decode activation must contain one token row"
                );
            }
            stage_result = state.session->decode_stage_activation(activation);
        }
        ++state.expected_decode_step;
        return inference::ExecutionResult::success(stage_output(std::move(stage_result)));
    }

    NativeStageConfig config;
    mutable std::mutex mutex;
    std::unordered_map<std::string, SessionState> sessions;
};

NativeStageExecutor::NativeStageExecutor(NativeStageConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

NativeStageExecutor::~NativeStageExecutor() = default;

inference::ExecutionResult NativeStageExecutor::execute(
    const inference::StageInput& input
) const {
    return impl_->execute(input);
}

void NativeStageExecutor::close_session(const std::string& session_id) const {
    impl_->close_session(session_id);
}

void NativeStageExecutor::rollback_session(
    const std::string& session_id,
    int token_count
) const {
    impl_->rollback_session(session_id, token_count);
}

std::size_t NativeStageExecutor::session_count() const {
    return impl_->session_count();
}

} // namespace jetsonfabric::runtime::adapters
