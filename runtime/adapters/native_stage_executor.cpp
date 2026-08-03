#include "adapters/native_stage_executor.hpp"

#include "inference/token_payload.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {

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
        if (config.position.index != 0 || config.position.count != 1) {
            throw std::invalid_argument("native serving currently requires one logical stage");
        }
        const int layer_count = static_cast<int>(config.engine->model_info().layer_count);
        if (config.layers.start != 0 || config.layers.end != layer_count) {
            throw std::invalid_argument("native serving currently requires the full model layer range");
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

    inference::ExecutionResult prefill(const inference::StageInput& input) {
        if (sessions.contains(input.session_id)) {
            return inference::ExecutionResult::invalid_input(
                "duplicate_stage_session",
                "prefill session already exists on this native stage"
            );
        }
        std::vector<std::int32_t> tokens = input_tokens(input);
        const std::size_t capacity = session_capacity(tokens.size(), input.max_tokens);
        std::unique_ptr<native::NativeSession> session =
            config.engine->create_session(capacity);
        const std::int32_t sampled = session->prefill_greedy(tokens);
        inference::StageOutput output = token_output(sampled);
        output.prompt_tokens = static_cast<int>(tokens.size());
        output.prompt_token_ids.reserve(tokens.size());
        for (const std::int32_t token : tokens) {
            if (token < 0) throw std::invalid_argument("native tokenizer returned a negative token");
            output.prompt_token_ids.push_back(static_cast<std::uint32_t>(token));
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
        const std::vector<std::int32_t> tokens = input_tokens(input);
        if (tokens.size() != 1U) {
            return inference::ExecutionResult::invalid_input(
                "native_multi_token_verification_unsupported",
                "native serving currently accepts one decode token per pass"
            );
        }
        const std::int32_t sampled = state.session->decode_greedy(tokens.front());
        ++state.expected_decode_step;
        return inference::ExecutionResult::success(token_output(sampled));
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
