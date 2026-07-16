#include "adapters/llama_cpp_stage_adapter.hpp"

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

void free_context(llama_context* context) {
    llama_free(context);
}

using ContextPtr = std::unique_ptr<llama_context, decltype(&free_context)>;

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

std::vector<std::int32_t> decode_tokens(const inference::Payload& payload) {
    if ((payload.tensor.dtype != "i32" && payload.tensor.dtype != "u32") ||
        payload.tensor.shape.size() != 1 || payload.bytes.size() % 4U != 0U) {
        throw std::invalid_argument("token payload must be a little-endian i32 or u32 vector");
    }
    std::vector<std::int32_t> tokens(payload.bytes.size() / 4U);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = read_i32_le(payload.bytes.data() + index * 4U);
    }
    return tokens;
}

std::int32_t decode_sampled_token(const inference::Payload& payload) {
    if ((payload.tensor.dtype != "i32" && payload.tensor.dtype != "u32") ||
        payload.tensor.shape != std::vector<std::int64_t>({1}) || payload.bytes.size() != 4U) {
        throw std::invalid_argument("sampled_token payload must contain one little-endian i32 or u32 token");
    }
    return read_i32_le(payload.bytes.data());
}

std::size_t activation_token_count(const inference::Payload& payload, int n_embd) {
    if (payload.tensor.dtype != "f32" || payload.tensor.shape.size() != 2 ||
        payload.tensor.shape[1] != n_embd || payload.tensor.shape[0] <= 0) {
        throw std::invalid_argument("activation payload must be f32[sequence_length, hidden_size]");
    }
    return static_cast<std::size_t>(payload.tensor.shape[0]);
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

inference::Payload sampled_token_payload(std::int32_t token) {
    inference::Payload payload;
    payload.kind = inference::PayloadKind::SampledToken;
    payload.tensor = inference::TensorDescriptor{
        .dtype = "i32",
        .shape = {1},
        .byte_order = "little",
        .layout = "row_major",
    };
    append_i32_le(payload.bytes, token);
    return payload;
}

void initialize_batch_common(llama_batch& batch, int32_t n_tokens, int32_t position, bool final_stage) {
    for (int32_t index = 0; index < n_tokens; ++index) {
        batch.pos[index] = position + index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = final_stage && index == n_tokens - 1;
    }
}

} // namespace

struct LlamaCppStageAdapter::Impl {
    struct Session {
        ContextPtr context{nullptr, free_context};
        int next_position = 0;
        int expected_decode_step = 1;
    };

    explicit Impl(LlamaCppStageConfig config_in)
        : config(std::move(config_in)) {
        if (!config.model) {
            throw std::invalid_argument("llama.cpp stage model is required");
        }
        if (config.position.count <= 0 || config.position.index < 0 ||
            config.position.index >= config.position.count) {
            throw std::invalid_argument("invalid llama.cpp stage position");
        }
        if (config.layers.start < 0 || config.layers.end <= config.layers.start ||
            config.layers.end > config.model->n_layer()) {
            throw std::invalid_argument("invalid llama.cpp stage layer range");
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
    }

    ContextPtr create_context() const {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(std::max(32, config.ctx_size));
        params.n_batch = params.n_ctx;
        params.n_ubatch = params.n_ctx;
        params.n_outputs_max = params.n_ctx;
        params.embeddings = !config.position.is_last();
        params.no_perf = true;
        if (config.threads > 0) {
            params.n_threads = config.threads;
            params.n_threads_batch = config.threads;
        }
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

    std::vector<std::int32_t> input_tokens(const inference::StageInput& input) const {
        if (input.payload.kind == inference::PayloadKind::Text) {
            return config.model->tokenize(
                std::string_view(
                    reinterpret_cast<const char*>(input.payload.bytes.data()),
                    input.payload.bytes.size()
                )
            );
        }
        if (input.payload.kind == inference::PayloadKind::Tokens) {
            return decode_tokens(input.payload);
        }
        if (input.payload.kind == inference::PayloadKind::SampledToken) {
            return {decode_sampled_token(input.payload)};
        }
        throw std::invalid_argument("first llama.cpp stage requires text, tokens, or sampled_token input");
    }

    inference::ExecutionResult run(Session& session, const inference::StageInput& input, bool prefill) const {
        const bool first_stage = config.position.is_first();
        const bool final_stage = config.position.is_last();
        int32_t n_tokens = 0;
        std::unique_ptr<BatchOwner> batch;

        if (first_stage) {
            const std::vector<std::int32_t> tokens = input_tokens(input);
            if (tokens.empty()) {
                return inference::ExecutionResult::invalid_input("empty_stage_tokens", "llama.cpp stage received no tokens");
            }
            n_tokens = static_cast<int32_t>(tokens.size());
            batch = std::make_unique<BatchOwner>(n_tokens, 0);
            for (int32_t index = 0; index < n_tokens; ++index) {
                batch->batch.token[index] = static_cast<llama_token>(tokens[static_cast<std::size_t>(index)]);
            }
        } else {
            const std::size_t token_count = activation_token_count(input.payload, config.model->n_embd());
            if (!prefill && token_count != 1U) {
                return inference::ExecutionResult::invalid_input(
                    "invalid_decode_activation",
                    "llama.cpp decode activation must contain exactly one token row"
                );
            }
            n_tokens = static_cast<int32_t>(token_count);
            batch = std::make_unique<BatchOwner>(n_tokens, config.model->n_embd());
            std::memcpy(batch->batch.embd, input.payload.bytes.data(), input.payload.bytes.size());
        }

        if (session.next_position + n_tokens > config.ctx_size) {
            return inference::ExecutionResult::invalid_input(
                "stage_context_exhausted",
                "llama.cpp stage context size would be exceeded"
            );
        }
        initialize_batch_common(batch->batch, n_tokens, session.next_position, final_stage);

        const int32_t status = llama_decode(session.context.get(), batch->batch);
        if (status != 0) {
            return inference::ExecutionResult::failure(
                "llama_stage_decode_failed",
                "llama_decode failed for assigned layer range with code " + std::to_string(status)
            );
        }

        session.next_position += n_tokens;
        if (!final_stage) {
            float* hidden = llama_get_embeddings(session.context.get());
            if (hidden == nullptr) {
                return inference::ExecutionResult::failure(
                    "llama_stage_activation_missing",
                    "llama.cpp did not expose the stage activation"
                );
            }
            inference::StageOutput output;
            output.payload = activation_payload(hidden, static_cast<std::size_t>(n_tokens), config.model->n_embd());
            if (prefill && first_stage) {
                output.prompt_tokens = n_tokens;
            }
            return inference::ExecutionResult::success(std::move(output));
        }

        float* logits = llama_get_logits_ith(session.context.get(), -1);
        if (logits == nullptr) {
            return inference::ExecutionResult::failure(
                "llama_stage_logits_missing",
                "llama.cpp did not expose final-stage logits"
            );
        }
        const int n_vocab = config.model->n_vocab();
        const auto best = std::max_element(logits, logits + n_vocab);
        const std::int32_t token = static_cast<std::int32_t>(std::distance(logits, best));
        inference::StageOutput output;
        output.payload = sampled_token_payload(token);
        output.end_of_generation = config.model->is_end_token(token);
        if (!output.end_of_generation) {
            output.token_text = config.model->token_piece(token);
            output.completion_tokens = 1;
        }
        return inference::ExecutionResult::success(std::move(output));
    }

    inference::ExecutionResult execute(const inference::StageInput& input) {
        std::lock_guard<std::mutex> lock(mutex);
        try {
            if (input.position.index != config.position.index || input.position.count != config.position.count ||
                input.layers.start != config.layers.start || input.layers.end != config.layers.end) {
                return inference::ExecutionResult::invalid_input(
                    "llama_stage_assignment_mismatch",
                    "stage input does not match the configured llama.cpp partition"
                );
            }

            if (input.phase == inference::Phase::Prefill) {
                if (sessions.find(input.session_id) != sessions.end()) {
                    return inference::ExecutionResult::invalid_input(
                        "duplicate_stage_session",
                        "prefill session already exists on this stage"
                    );
                }
                Session session;
                session.context = create_context();
                inference::ExecutionResult result = run(session, input, true);
                if (result.ok) {
                    sessions.emplace(input.session_id, std::move(session));
                }
                return result;
            }

            auto found = sessions.find(input.session_id);
            if (found == sessions.end()) {
                return inference::ExecutionResult::invalid_input(
                    "stage_session_not_found",
                    "decode requires an existing prefill session"
                );
            }
            Session& session = found->second;
            if (input.decode_step != session.expected_decode_step) {
                return inference::ExecutionResult::invalid_input(
                    "decode_step_mismatch",
                    "decode_step does not match the stage session"
                );
            }
            inference::ExecutionResult result = run(session, input, false);
            if (result.ok) {
                session.expected_decode_step += 1;
            }
            return result;
        } catch (const std::invalid_argument& error) {
            return inference::ExecutionResult::invalid_input("invalid_llama_stage_input", error.what());
        } catch (const std::exception& error) {
            return inference::ExecutionResult::failure("llama_stage_execution_failed", error.what());
        }
    }

    LlamaCppStageConfig config;
    std::mutex mutex;
    std::unordered_map<std::string, Session> sessions;
};

LlamaCppStageAdapter::LlamaCppStageAdapter(LlamaCppStageConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LlamaCppStageAdapter::~LlamaCppStageAdapter() = default;

inference::ExecutionResult LlamaCppStageAdapter::execute(const inference::StageInput& input) const {
    return impl_->execute(input);
}

void LlamaCppStageAdapter::close_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->sessions.erase(session_id);
}

std::size_t LlamaCppStageAdapter::session_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->sessions.size();
}

} // namespace jetsonfabric::runtime::adapters
