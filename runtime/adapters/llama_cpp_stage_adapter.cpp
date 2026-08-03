#include "adapters/llama_cpp_stage_adapter.hpp"

#include "inference/token_payload.hpp"

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

void free_context(llama_context* context) {
    if (context == nullptr) {
        return;
    }
    llama_synchronize(context);
    llama_free(context);
}

using ContextPtr = std::unique_ptr<llama_context, decltype(&free_context)>;
using SessionClock = std::chrono::steady_clock;

ggml_type llama_kv_cache_type(KVCacheType type) {
    switch (type) {
    case KVCacheType::F16:
        return GGML_TYPE_F16;
    case KVCacheType::Q8_0:
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_F16;
}

struct BatchOwner {
    llama_batch batch{};

    BatchOwner(int32_t n_tokens, int32_t embedding_size)
        : batch(llama_batch_init(n_tokens, embedding_size, 1)) {
        batch.n_tokens = n_tokens;
    }

    ~BatchOwner() {
        llama_batch_free(batch);
    }

    BatchOwner(const BatchOwner&) = delete;
    BatchOwner& operator=(const BatchOwner&) = delete;
};

std::size_t activation_token_count(const inference::Payload& payload, int n_embd) {
    if (payload.tensor.dtype != "f32" || payload.tensor.shape.size() != 2 ||
        payload.tensor.shape[1] != n_embd || payload.tensor.shape[0] <= 0) {
        throw std::invalid_argument("activation payload must be f32[sequence_length, hidden_size]");
    }
    const std::size_t token_count = static_cast<std::size_t>(payload.tensor.shape[0]);
    const std::size_t expected_bytes = token_count * static_cast<std::size_t>(n_embd) * sizeof(float);
    if (payload.bytes.size() != expected_bytes) {
        throw std::invalid_argument("activation payload byte count does not match its tensor shape");
    }
    return token_count;
}

inference::Payload activation_payload(const float* values, std::size_t token_count, int n_embd) {
    const std::size_t byte_count = token_count * static_cast<std::size_t>(n_embd) * sizeof(float);
    inference::Payload payload;
    payload.kind = inference::PayloadKind::Activation;
    payload.tensor = inference::TensorDescriptor{
        .dtype = "f32",
        .shape = {static_cast<std::int64_t>(token_count), n_embd},
        .byte_order = "little",
        .layout = "row_major",
    };
    payload.bytes.resize(byte_count);
    std::memcpy(payload.bytes.data(), values, byte_count);
    return payload;
}

inference::Payload activation_payload(std::vector<std::uint8_t> bytes, std::size_t token_count, int n_embd) {
    inference::Payload payload;
    payload.kind = inference::PayloadKind::Activation;
    payload.tensor = inference::TensorDescriptor{
        .dtype = "f32",
        .shape = {static_cast<std::int64_t>(token_count), n_embd},
        .byte_order = "little",
        .layout = "row_major",
    };
    payload.bytes = std::move(bytes);
    return payload;
}

void initialize_batch_common(
    llama_batch& batch,
    int32_t n_tokens,
    int32_t position,
    llama_seq_id sequence_id,
    bool final_stage,
    bool all_outputs = false
) {
    for (int32_t index = 0; index < n_tokens; ++index) {
        batch.pos[index] = position + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence_id;
        batch.logits[index] = final_stage && (all_outputs || index == n_tokens - 1);
    }
}

inference::ExecutionResult decode_failure(int status) {
    return inference::ExecutionResult::failure(
        "llama_stage_decode_failed",
        "llama_decode failed for assigned layer range with code " + std::to_string(status)
    );
}

} // namespace

struct LlamaCppStageAdapter::Impl {
    struct SerialSession {
        ContextPtr context{nullptr, free_context};
        int next_position = 0;
        int expected_decode_step = 1;
        SessionClock::time_point last_used = SessionClock::now();
    };

    struct SharedSession {
        llama_seq_id sequence_id = 0;
        int next_position = 0;
        int expected_decode_step = 1;
        SessionClock::time_point last_used = SessionClock::now();
    };

    explicit Impl(LlamaCppStageConfig config_in)
        : config(std::move(config_in)) {
        validate_config();
        shared_sequence_in_use.resize(
            static_cast<std::size_t>(config.max_parallel_sessions),
            false
        );
        reaper = std::jthread([this](std::stop_token stop_token) {
            reap_sessions(stop_token);
        });
    }

    void validate_config() const {
        if (!config.model) {
            throw std::invalid_argument("llama.cpp stage model is required");
        }
        if (config.ubatch_size <= 0) {
            throw std::invalid_argument("llama.cpp stage micro-batch size must be greater than zero");
        }
        if (config.decode_batch_size <= 0 ||
            config.decode_batch_size > config.max_parallel_sessions) {
            throw std::invalid_argument(
                "llama.cpp decode batch size must be between one and max parallel sessions"
            );
        }
        if (config.position.count <= 0 || config.position.index < 0 ||
            config.position.index >= config.position.count) {
            throw std::invalid_argument("invalid llama.cpp stage position");
        }
        if (config.layers.start < 0 || config.layers.end <= config.layers.start ||
            config.layers.end > config.model->n_layer()) {
            throw std::invalid_argument("invalid llama.cpp stage layer range");
        }
        if (config.layers.start < config.model->loaded_layer_start() ||
            config.layers.end > config.model->loaded_layer_end()) {
            throw std::invalid_argument("llama.cpp stage range exceeds resident model layers");
        }
        if (config.position.is_first() && config.layers.start != 0) {
            throw std::invalid_argument("first llama.cpp stage must start at layer zero");
        }
        if (config.position.is_last() && config.layers.end != config.model->n_layer()) {
            throw std::invalid_argument("final llama.cpp stage must end at the model layer count");
        }
        if (config.model->architecture() != "llama" && config.model->architecture() != "qwen2") {
            throw std::invalid_argument(
                "partial-layer execution currently supports llama and qwen2 architectures; got " +
                config.model->architecture()
            );
        }
        if (config.session_idle_ttl <= std::chrono::milliseconds::zero() ||
            config.session_reap_interval <= std::chrono::milliseconds::zero()) {
            throw std::invalid_argument("llama.cpp session expiration intervals must be greater than zero");
        }
    }

    bool batching_enabled() const noexcept {
        return config.decode_batch_size > 1;
    }

    void apply_context_options(llama_context_params& params) const {
        params.embeddings = !config.position.is_last();
        params.no_perf = true;
        params.type_k = llama_kv_cache_type(config.kv_cache_type);
        params.type_v = llama_kv_cache_type(config.kv_cache_type);
        if (config.threads > 0) {
            params.n_threads = config.threads;
            params.n_threads_batch = config.threads;
        }
    }

    ContextPtr initialize_context(llama_context_params params) const {
        llama_context* raw = llama_init_from_model(config.model->raw_model(), params);
        if (raw == nullptr) {
            throw std::runtime_error("failed to create llama.cpp stage context");
        }
        ContextPtr context(raw, free_context);
        if (!llama_set_layer_range(
                context.get(),
                static_cast<std::uint32_t>(config.layers.start),
                static_cast<std::uint32_t>(config.layers.end))) {
            throw std::runtime_error("llama.cpp rejected the configured stage layer range");
        }
        return context;
    }

    ContextPtr create_serial_context(int prefill_tokens) const {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(std::max(32, config.ctx_size));
        const std::uint32_t batch_size = static_cast<std::uint32_t>(std::clamp(
            std::max(prefill_tokens, config.speculative_max_tokens + 1),
            1,
            static_cast<int>(params.n_ctx)
        ));
        params.n_batch = batch_size;
        params.n_ubatch = std::min(batch_size, static_cast<std::uint32_t>(config.ubatch_size));
        params.n_outputs_max = config.position.is_last()
            ? static_cast<std::uint32_t>(std::max(1, config.speculative_max_tokens + 1))
            : batch_size;
        apply_context_options(params);
        return initialize_context(params);
    }

    ContextPtr create_shared_context() const {
        const std::uint64_t total_context =
            static_cast<std::uint64_t>(std::max(32, config.ctx_size)) *
            static_cast<std::uint64_t>(config.max_parallel_sessions);
        if (total_context > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("parallel llama.cpp context exceeds uint32 capacity");
        }

        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(total_context);
        params.n_batch = static_cast<std::uint32_t>(std::min(
            config.ctx_size,
            std::max(config.ubatch_size, config.decode_batch_size)
        ));
        params.n_ubatch = std::min(
            params.n_batch,
            static_cast<std::uint32_t>(config.ubatch_size)
        );
        params.n_seq_max = static_cast<std::uint32_t>(config.max_parallel_sessions);
        params.n_outputs_max = config.position.is_last()
            ? static_cast<std::uint32_t>(std::max(
                  config.decode_batch_size,
                  config.max_parallel_sessions
              ))
            : params.n_batch;
        apply_context_options(params);
        return initialize_context(params);
    }

    std::vector<std::int32_t> input_tokens(const inference::StageInput& input) const {
        if (input.payload.kind == inference::PayloadKind::Text) {
            return config.model->tokenize(std::string_view(
                reinterpret_cast<const char*>(input.payload.bytes.data()),
                input.payload.bytes.size()
            ));
        }
        if (input.payload.kind == inference::PayloadKind::Tokens) {
            return inference::decode_token_ids(input.payload);
        }
        if (input.payload.kind == inference::PayloadKind::SampledToken) {
            return {inference::decode_single_token(input.payload)};
        }
        throw std::invalid_argument("first llama.cpp stage requires text, tokens, or sampled_token input");
    }

    static std::vector<std::uint32_t> prompt_token_ids(
        const std::vector<std::int32_t>& tokens
    ) {
        std::vector<std::uint32_t> ids;
        ids.reserve(tokens.size());
        for (const std::int32_t token : tokens) {
            if (token < 0) {
                throw std::runtime_error("llama.cpp returned a negative prompt token ID");
            }
            ids.push_back(static_cast<std::uint32_t>(token));
        }
        return ids;
    }

    bool assignment_matches(const inference::StageInput& input) const noexcept {
        return input.position.index == config.position.index &&
            input.position.count == config.position.count &&
            input.layers.start == config.layers.start &&
            input.layers.end == config.layers.end;
    }

    inference::ExecutionResult output_for_row(
        llama_context* context,
        int row,
        int execution_batch_size
    ) const {
        inference::StageOutput output;
        output.execution_batch_size = execution_batch_size;
        if (!config.position.is_last()) {
            float* hidden = llama_get_embeddings_ith(context, row);
            if (hidden == nullptr) {
                return inference::ExecutionResult::failure(
                    "llama_stage_activation_missing",
                    "llama.cpp did not expose the stage activation"
                );
            }
            output.payload = activation_payload(hidden, 1U, config.model->n_embd());
            return inference::ExecutionResult::success(std::move(output));
        }

        float* logits = llama_get_logits_ith(context, row);
        if (logits == nullptr) {
            return inference::ExecutionResult::failure(
                "llama_stage_logits_missing",
                "llama.cpp did not expose final-stage logits"
            );
        }
        const int n_vocab = config.model->n_vocab();
        const auto best = std::max_element(logits, logits + n_vocab);
        const std::int32_t token = static_cast<std::int32_t>(std::distance(logits, best));
        output.payload = inference::sampled_token_payload(token);
        output.end_of_generation = config.model->is_end_token(token);
        if (!output.end_of_generation) {
            output.token_text = config.model->token_piece(token);
            output.completion_tokens = 1;
        }
        return inference::ExecutionResult::success(std::move(output));
    }

    inference::ExecutionResult output_for_rows(
        llama_context* context,
        int row_count
    ) const {
        inference::StageOutput output;
        output.execution_batch_size = 1;
        output.verification_width = row_count;
        std::vector<std::int32_t> tokens;
        tokens.reserve(static_cast<std::size_t>(row_count));
        output.token_text_offsets.reserve(static_cast<std::size_t>(row_count));
        output.token_eog.reserve(static_cast<std::size_t>(row_count));

        const int n_vocab = config.model->n_vocab();
        for (int row = 0; row < row_count; ++row) {
            float* logits = llama_get_logits_ith(context, row);
            if (logits == nullptr) {
                return inference::ExecutionResult::failure(
                    "llama_stage_logits_missing",
                    "llama.cpp did not expose speculative verification logits"
                );
            }
            const auto best = std::max_element(logits, logits + n_vocab);
            const std::int32_t token = static_cast<std::int32_t>(std::distance(logits, best));
            const bool end_of_generation = config.model->is_end_token(token);
            tokens.push_back(token);
            output.token_eog.push_back(end_of_generation ? 1U : 0U);
            if (!end_of_generation) {
                output.token_text += config.model->token_piece(token);
                ++output.completion_tokens;
            }
            output.token_text_offsets.push_back(
                static_cast<std::uint32_t>(output.token_text.size())
            );
        }
        output.payload = inference::sampled_tokens_payload(tokens);
        return inference::ExecutionResult::success(std::move(output));
    }

    inference::ExecutionResult run_serial(
        SerialSession& session,
        const inference::StageInput& input,
        bool prefill
    ) const {
        const bool first_stage = config.position.is_first();
        int32_t n_tokens = 0;
        std::unique_ptr<BatchOwner> batch;
        std::vector<std::int32_t> tokens;

        if (first_stage) {
            tokens = input_tokens(input);
            if (tokens.empty()) {
                return inference::ExecutionResult::invalid_input(
                    "empty_stage_tokens",
                    "llama.cpp stage received no tokens"
                );
            }
            n_tokens = static_cast<int32_t>(tokens.size());
            batch = std::make_unique<BatchOwner>(n_tokens, 0);
            for (int32_t index = 0; index < n_tokens; ++index) {
                batch->batch.token[index] = tokens[static_cast<std::size_t>(index)];
            }
        } else {
            const std::size_t token_count = activation_token_count(input.payload, config.model->n_embd());
            const std::size_t max_decode_rows = static_cast<std::size_t>(
                std::max(1, config.speculative_max_tokens + 1)
            );
            if (!prefill && (token_count == 0U || token_count > max_decode_rows)) {
                return inference::ExecutionResult::invalid_input(
                    "invalid_decode_activation",
                    "llama.cpp decode activation exceeds the configured verification width"
                );
            }
            n_tokens = static_cast<int32_t>(token_count);
            batch = std::make_unique<BatchOwner>(n_tokens, config.model->n_embd());
            std::memcpy(batch->batch.embd, input.payload.bytes.data(), input.payload.bytes.size());
        }

        if (context_exhausted(session.next_position, n_tokens, input)) {
            return context_exhausted_result();
        }
        if (!session.context) {
            session.context = create_serial_context(n_tokens);
        }
        initialize_batch_common(
            batch->batch,
            n_tokens,
            session.next_position,
            0,
            config.position.is_last(),
            !prefill
        );

        const int32_t status = llama_decode(session.context.get(), batch->batch);
        if (status != 0) {
            return decode_failure(status);
        }
        session.next_position += n_tokens;

        if (!config.position.is_last()) {
            float* hidden = llama_get_embeddings(session.context.get());
            if (hidden == nullptr) {
                return inference::ExecutionResult::failure(
                    "llama_stage_activation_missing",
                    "llama.cpp did not expose the stage activation"
                );
            }
            inference::StageOutput output;
            output.payload = activation_payload(hidden, static_cast<std::size_t>(n_tokens), config.model->n_embd());
            output.execution_batch_size = 1;
            output.verification_width = prefill ? 1 : n_tokens;
            if (prefill && first_stage) {
                output.prompt_tokens = n_tokens;
                output.prompt_token_ids = prompt_token_ids(tokens);
            }
            return inference::ExecutionResult::success(std::move(output));
        }

        inference::ExecutionResult result = !prefill && n_tokens > 1
            ? output_for_rows(session.context.get(), n_tokens)
            : output_for_row(session.context.get(), -1, 1);
        if (result.ok && prefill && first_stage) {
            result.output.prompt_tokens = n_tokens;
            result.output.prompt_token_ids = prompt_token_ids(tokens);
        }
        return result;
    }

    bool context_exhausted(
        int next_position,
        int token_count,
        const inference::StageInput& input
    ) const noexcept {
        const int requested_decode_positions = input.phase == inference::Phase::Prefill
            ? std::max(0, input.max_tokens - 1)
            : 0;
        return next_position + token_count + requested_decode_positions > config.ctx_size;
    }

    static inference::ExecutionResult context_exhausted_result() {
        return inference::ExecutionResult::invalid_input(
            "stage_context_exhausted",
            "prompt and requested completion exceed the llama.cpp stage context size"
        );
    }

    inference::ExecutionResult execute_serial_locked(const inference::StageInput& input) {
        if (!assignment_matches(input)) {
            return assignment_mismatch_result();
        }
        if (input.phase == inference::Phase::Prefill) {
            if (serial_sessions.contains(input.session_id)) {
                return duplicate_session_result();
            }
            SerialSession session;
            inference::ExecutionResult result = run_serial(session, input, true);
            if (result.ok) {
                session.last_used = SessionClock::now();
                serial_sessions.emplace(input.session_id, std::move(session));
            }
            return result;
        }

        auto found = serial_sessions.find(input.session_id);
        if (found == serial_sessions.end()) {
            return missing_session_result();
        }
        SerialSession& session = found->second;
        if (input.decode_step != session.expected_decode_step) {
            return decode_step_mismatch_result();
        }
        inference::ExecutionResult result = run_serial(session, input, false);
        if (result.ok) {
            const int decoded_tokens = input.payload.kind == inference::PayloadKind::Activation
                ? static_cast<int>(input.payload.tensor.shape.front())
                : static_cast<int>(input.payload.bytes.size() / sizeof(std::int32_t));
            session.expected_decode_step += decoded_tokens;
            session.last_used = SessionClock::now();
        }
        return result;
    }

    llama_seq_id allocate_sequence() {
        for (std::size_t index = 0; index < shared_sequence_in_use.size(); ++index) {
            if (!shared_sequence_in_use[index]) {
                shared_sequence_in_use[index] = true;
                return static_cast<llama_seq_id>(index);
            }
        }
        return -1;
    }

    void release_sequence(llama_seq_id sequence_id) {
        if (shared_context) {
            (void) llama_memory_seq_rm(
                llama_get_memory(shared_context.get()),
                sequence_id,
                -1,
                -1
            );
        }
        if (sequence_id >= 0 &&
            static_cast<std::size_t>(sequence_id) < shared_sequence_in_use.size()) {
            shared_sequence_in_use[static_cast<std::size_t>(sequence_id)] = false;
        }
    }

    inference::ExecutionResult run_shared_prefill(
        SharedSession& session,
        const inference::StageInput& input
    ) {
        const bool first_stage = config.position.is_first();
        const std::vector<std::int32_t> tokens = first_stage
            ? input_tokens(input)
            : std::vector<std::int32_t>{};
        const std::size_t token_count = first_stage
            ? tokens.size()
            : activation_token_count(input.payload, config.model->n_embd());
        if (token_count == 0U) {
            return inference::ExecutionResult::invalid_input(
                "empty_stage_tokens",
                "llama.cpp stage received no tokens"
            );
        }
        if (context_exhausted(session.next_position, static_cast<int>(token_count), input)) {
            return context_exhausted_result();
        }
        if (!shared_context) {
            shared_context = create_shared_context();
        }

        std::vector<std::uint8_t> activation_bytes;
        if (!config.position.is_last()) {
            activation_bytes.reserve(
                token_count * static_cast<std::size_t>(config.model->n_embd()) * sizeof(float)
            );
        }
        const std::size_t chunk_limit = static_cast<std::size_t>(llama_n_batch(shared_context.get()));
        for (std::size_t offset = 0; offset < token_count; offset += chunk_limit) {
            const std::size_t chunk_size = std::min(chunk_limit, token_count - offset);
            const bool final_chunk = offset + chunk_size == token_count;
            const int status = decode_shared_prefill_chunk(
                session,
                input,
                tokens,
                offset,
                chunk_size,
                final_chunk,
                activation_bytes
            );
            if (status != 0) {
                return decode_failure(status);
            }
        }
        session.next_position += static_cast<int>(token_count);

        if (!config.position.is_last()) {
            inference::StageOutput output;
            output.payload = activation_payload(
                std::move(activation_bytes),
                token_count,
                config.model->n_embd()
            );
            output.execution_batch_size = 1;
            if (first_stage) {
                output.prompt_tokens = static_cast<int>(token_count);
                output.prompt_token_ids = prompt_token_ids(tokens);
            }
            return inference::ExecutionResult::success(std::move(output));
        }

        inference::ExecutionResult result = output_for_row(shared_context.get(), -1, 1);
        if (result.ok && first_stage) {
            result.output.prompt_tokens = static_cast<int>(token_count);
            result.output.prompt_token_ids = prompt_token_ids(tokens);
        }
        return result;
    }

    int decode_shared_prefill_chunk(
        const SharedSession& session,
        const inference::StageInput& input,
        const std::vector<std::int32_t>& tokens,
        std::size_t offset,
        std::size_t chunk_size,
        bool final_chunk,
        std::vector<std::uint8_t>& activation_bytes
    ) const {
        BatchOwner batch(
            static_cast<int32_t>(chunk_size),
            config.position.is_first() ? 0 : config.model->n_embd()
        );
        if (config.position.is_first()) {
            for (std::size_t index = 0; index < chunk_size; ++index) {
                batch.batch.token[index] = tokens[offset + index];
            }
        } else {
            const std::size_t row_bytes = static_cast<std::size_t>(config.model->n_embd()) * sizeof(float);
            std::memcpy(
                batch.batch.embd,
                input.payload.bytes.data() + offset * row_bytes,
                chunk_size * row_bytes
            );
        }
        initialize_batch_common(
            batch.batch,
            static_cast<int32_t>(chunk_size),
            session.next_position + static_cast<int>(offset),
            session.sequence_id,
            config.position.is_last() && final_chunk
        );
        const int status = llama_decode(shared_context.get(), batch.batch);
        if (status != 0 || config.position.is_last()) {
            return status;
        }

        const std::size_t row_bytes = static_cast<std::size_t>(config.model->n_embd()) * sizeof(float);
        for (std::size_t index = 0; index < chunk_size; ++index) {
            float* hidden = llama_get_embeddings_ith(shared_context.get(), static_cast<int32_t>(index));
            if (hidden == nullptr) {
                throw std::runtime_error("llama.cpp did not expose a prefill activation row");
            }
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(hidden);
            activation_bytes.insert(activation_bytes.end(), bytes, bytes + row_bytes);
        }
        return 0;
    }

    inference::ExecutionResult execute_shared_prefill_locked(const inference::StageInput& input) {
        if (!assignment_matches(input)) {
            return assignment_mismatch_result();
        }
        if (shared_sessions.contains(input.session_id)) {
            return duplicate_session_result();
        }
        const llama_seq_id sequence_id = allocate_sequence();
        if (sequence_id < 0) {
            return inference::ExecutionResult::failure(
                "stage_session_capacity_exceeded",
                "all configured llama.cpp parallel session slots are in use"
            );
        }

        SharedSession session{.sequence_id = sequence_id};
        inference::ExecutionResult result = run_shared_prefill(session, input);
        if (!result.ok) {
            release_sequence(sequence_id);
            return result;
        }
        session.last_used = SessionClock::now();
        shared_sessions.emplace(input.session_id, session);
        return result;
    }

    std::vector<inference::ExecutionResult> execute_shared_decode_locked(
        const std::vector<inference::StageInput>& inputs
    ) {
        std::vector<inference::ExecutionResult> results(inputs.size());
        std::vector<std::size_t> valid;
        valid.reserve(inputs.size());
        std::unordered_set<llama_seq_id> sequences;

        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const inference::StageInput& input = inputs[index];
            if (!assignment_matches(input)) {
                results[index] = assignment_mismatch_result();
                continue;
            }
            auto found = shared_sessions.find(input.session_id);
            if (found == shared_sessions.end()) {
                results[index] = missing_session_result();
                continue;
            }
            if (input.decode_step != found->second.expected_decode_step) {
                results[index] = decode_step_mismatch_result();
                continue;
            }
            if (!sequences.insert(found->second.sequence_id).second) {
                results[index] = inference::ExecutionResult::invalid_input(
                    "duplicate_batch_session",
                    "a decode batch cannot contain the same session twice"
                );
                continue;
            }
            if (context_exhausted(found->second.next_position, 1, input)) {
                results[index] = context_exhausted_result();
                continue;
            }
            valid.push_back(index);
        }
        if (valid.empty()) {
            return results;
        }
        if (valid.size() > static_cast<std::size_t>(config.decode_batch_size)) {
            return batch_too_large_results(inputs.size());
        }

        try {
            decode_shared_rows(inputs, valid, results);
        } catch (const std::invalid_argument& error) {
            for (const std::size_t index : valid) {
                results[index] = inference::ExecutionResult::invalid_input(
                    "invalid_llama_stage_input",
                    error.what()
                );
            }
        } catch (const std::exception& error) {
            for (const std::size_t index : valid) {
                results[index] = inference::ExecutionResult::failure(
                    "llama_stage_execution_failed",
                    error.what()
                );
            }
        }
        return results;
    }

    void decode_shared_rows(
        const std::vector<inference::StageInput>& inputs,
        const std::vector<std::size_t>& valid,
        std::vector<inference::ExecutionResult>& results
    ) {
        if (!shared_context) {
            throw std::runtime_error("shared llama.cpp context is missing after prefill");
        }
        BatchOwner batch(
            static_cast<int32_t>(valid.size()),
            config.position.is_first() ? 0 : config.model->n_embd()
        );
        const std::size_t activation_row_bytes =
            static_cast<std::size_t>(config.model->n_embd()) * sizeof(float);
        for (std::size_t row = 0; row < valid.size(); ++row) {
            const inference::StageInput& input = inputs[valid[row]];
            SharedSession& session = shared_sessions.at(input.session_id);
            if (config.position.is_first()) {
                batch.batch.token[row] = inference::decode_single_token(input.payload);
            } else {
                if (activation_token_count(input.payload, config.model->n_embd()) != 1U) {
                    throw std::invalid_argument(
                        "llama.cpp decode activation must contain exactly one token row"
                    );
                }
                std::memcpy(
                    batch.batch.embd + row * static_cast<std::size_t>(config.model->n_embd()),
                    input.payload.bytes.data(),
                    activation_row_bytes
                );
            }
            batch.batch.pos[row] = session.next_position;
            batch.batch.n_seq_id[row] = 1;
            batch.batch.seq_id[row][0] = session.sequence_id;
            batch.batch.logits[row] = config.position.is_last();
        }

        const int status = llama_decode(shared_context.get(), batch.batch);
        if (status != 0) {
            for (const std::size_t index : valid) {
                results[index] = decode_failure(status);
            }
            return;
        }

        const int batch_size = static_cast<int>(valid.size());
        for (std::size_t row = 0; row < valid.size(); ++row) {
            const std::size_t index = valid[row];
            results[index] = output_for_row(shared_context.get(), static_cast<int>(row), batch_size);
            if (!results[index].ok) {
                continue;
            }
            SharedSession& session = shared_sessions.at(inputs[index].session_id);
            ++session.next_position;
            ++session.expected_decode_step;
            session.last_used = SessionClock::now();
        }
    }

    std::vector<inference::ExecutionResult> execute_shared_locked(
        const std::vector<inference::StageInput>& inputs
    ) {
        if (inputs.empty()) {
            return {};
        }
        const bool all_decode = std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
            return input.phase == inference::Phase::Decode;
        });
        if (all_decode) {
            return execute_shared_decode_locked(inputs);
        }

        std::vector<inference::ExecutionResult> results;
        results.reserve(inputs.size());
        for (const inference::StageInput& input : inputs) {
            results.push_back(
                input.phase == inference::Phase::Prefill
                    ? execute_shared_prefill_locked(input)
                    : execute_shared_decode_locked({input}).front()
            );
        }
        return results;
    }

    inference::ExecutionResult execute(const inference::StageInput& input) {
        return execute_batch({input}).front();
    }

    std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>& inputs
    ) {
        const std::lock_guard lock(mutex);
        try {
            prune_expired_sessions_locked(SessionClock::now());
            if (batching_enabled()) {
                return execute_shared_locked(inputs);
            }
            std::vector<inference::ExecutionResult> results;
            results.reserve(inputs.size());
            for (const inference::StageInput& input : inputs) {
                results.push_back(execute_serial_locked(input));
            }
            return results;
        } catch (const std::invalid_argument& error) {
            return repeated_failure(
                inputs.size(),
                inference::ExecutionResult::invalid_input("invalid_llama_stage_input", error.what())
            );
        } catch (const std::exception& error) {
            return repeated_failure(
                inputs.size(),
                inference::ExecutionResult::failure("llama_stage_execution_failed", error.what())
            );
        }
    }

    void close_session_locked(const std::string& session_id) {
        if (!batching_enabled()) {
            serial_sessions.erase(session_id);
            return;
        }
        const auto found = shared_sessions.find(session_id);
        if (found == shared_sessions.end()) {
            return;
        }
        release_sequence(found->second.sequence_id);
        shared_sessions.erase(found);
    }

    void rollback_session_locked(const std::string& session_id, int token_count) {
        if (token_count <= 0) {
            throw std::invalid_argument("rollback token count must be greater than zero");
        }
        if (!batching_enabled()) {
            const auto found = serial_sessions.find(session_id);
            if (found == serial_sessions.end()) throw std::invalid_argument("stage session not found");
            SerialSession& session = found->second;
            if (token_count > session.next_position || token_count >= session.expected_decode_step) {
                throw std::invalid_argument("rollback exceeds the decoded session suffix");
            }
            const int rollback_position = session.next_position - token_count;
            if (!llama_memory_seq_rm(
                    llama_get_memory(session.context.get()),
                    0,
                    rollback_position,
                    -1)) {
                throw std::runtime_error("llama.cpp rejected session rollback");
            }
            session.next_position = rollback_position;
            session.expected_decode_step -= token_count;
            session.last_used = SessionClock::now();
            return;
        }

        const auto found = shared_sessions.find(session_id);
        if (found == shared_sessions.end()) throw std::invalid_argument("stage session not found");
        SharedSession& session = found->second;
        if (token_count > session.next_position || token_count >= session.expected_decode_step) {
            throw std::invalid_argument("rollback exceeds the decoded session suffix");
        }
        const int rollback_position = session.next_position - token_count;
        if (!llama_memory_seq_rm(
                llama_get_memory(shared_context.get()),
                session.sequence_id,
                rollback_position,
                -1)) {
            throw std::runtime_error("llama.cpp rejected session rollback");
        }
        session.next_position = rollback_position;
        session.expected_decode_step -= token_count;
        session.last_used = SessionClock::now();
    }

    void prune_expired_sessions_locked(SessionClock::time_point now) {
        if (!batching_enabled()) {
            for (auto found = serial_sessions.begin(); found != serial_sessions.end();) {
                found = now - found->second.last_used >= config.session_idle_ttl
                    ? serial_sessions.erase(found)
                    : std::next(found);
            }
            return;
        }
        for (auto found = shared_sessions.begin(); found != shared_sessions.end();) {
            if (now - found->second.last_used >= config.session_idle_ttl) {
                release_sequence(found->second.sequence_id);
                found = shared_sessions.erase(found);
            } else {
                ++found;
            }
        }
    }

    void reap_sessions(std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            std::this_thread::sleep_for(config.session_reap_interval);
            if (stop_token.stop_requested()) {
                return;
            }
            const std::lock_guard lock(mutex);
            prune_expired_sessions_locked(SessionClock::now());
        }
    }

    std::size_t session_count_locked() const noexcept {
        return batching_enabled() ? shared_sessions.size() : serial_sessions.size();
    }

    static inference::ExecutionResult assignment_mismatch_result() {
        return inference::ExecutionResult::invalid_input(
            "llama_stage_assignment_mismatch",
            "stage input does not match the configured llama.cpp partition"
        );
    }

    static inference::ExecutionResult duplicate_session_result() {
        return inference::ExecutionResult::invalid_input(
            "duplicate_stage_session",
            "prefill session already exists on this stage"
        );
    }

    static inference::ExecutionResult missing_session_result() {
        return inference::ExecutionResult::invalid_input(
            "stage_session_not_found",
            "decode requires an existing non-expired prefill session"
        );
    }

    static inference::ExecutionResult decode_step_mismatch_result() {
        return inference::ExecutionResult::invalid_input(
            "decode_step_mismatch",
            "decode_step does not match the stage session"
        );
    }

    static std::vector<inference::ExecutionResult> batch_too_large_results(std::size_t count) {
        return repeated_failure(
            count,
            inference::ExecutionResult::invalid_input(
                "decode_batch_too_large",
                "decode batch exceeds the configured maximum"
            )
        );
    }

    static std::vector<inference::ExecutionResult> repeated_failure(
        std::size_t count,
        const inference::ExecutionResult& result
    ) {
        return std::vector<inference::ExecutionResult>(count, result);
    }

    LlamaCppStageConfig config;
    mutable std::mutex mutex;
    std::unordered_map<std::string, SerialSession> serial_sessions;
    ContextPtr shared_context{nullptr, free_context};
    std::unordered_map<std::string, SharedSession> shared_sessions;
    std::vector<bool> shared_sequence_in_use;
    std::jthread reaper;
};

LlamaCppStageAdapter::LlamaCppStageAdapter(LlamaCppStageConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LlamaCppStageAdapter::~LlamaCppStageAdapter() = default;

inference::ExecutionResult LlamaCppStageAdapter::execute(const inference::StageInput& input) const {
    return impl_->execute(input);
}

std::vector<inference::ExecutionResult> LlamaCppStageAdapter::execute_batch(
    const std::vector<inference::StageInput>& inputs
) const {
    return impl_->execute_batch(inputs);
}

void LlamaCppStageAdapter::close_session(const std::string& session_id) const {
    const std::lock_guard lock(impl_->mutex);
    impl_->close_session_locked(session_id);
}

void LlamaCppStageAdapter::rollback_session(const std::string& session_id, int token_count) const {
    const std::lock_guard lock(impl_->mutex);
    impl_->rollback_session_locked(session_id, token_count);
}

std::size_t LlamaCppStageAdapter::session_count() const {
    const std::lock_guard lock(impl_->mutex);
    impl_->prune_expired_sessions_locked(SessionClock::now());
    return impl_->session_count_locked();
}

} // namespace jetsonfabric::runtime::adapters
