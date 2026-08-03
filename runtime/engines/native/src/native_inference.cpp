#include "jetsonfabric/native_inference.hpp"

#include "model_architecture.hpp"
#include "tensor_store.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::native {

class NativeSession::Impl {
public:
    explicit Impl(std::unique_ptr<InferenceSession> session_in)
        : session(std::move(session_in)) {
        if (!session) throw std::invalid_argument("native session is required");
    }

    std::unique_ptr<InferenceSession> session;
};

NativeSession::NativeSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
NativeSession::~NativeSession() = default;
NativeSession::NativeSession(NativeSession&&) noexcept = default;
NativeSession& NativeSession::operator=(NativeSession&&) noexcept = default;

std::size_t NativeSession::capacity() const { return impl_->session->capacity(); }
std::size_t NativeSession::position() const { return impl_->session->position(); }

std::int32_t NativeSession::prefill_greedy(std::span<const std::int32_t> tokens) {
    return impl_->session->prefill_greedy(tokens).token;
}

std::int32_t NativeSession::decode_greedy(std::int32_t token) {
    return impl_->session->decode_greedy(token);
}

StageResult NativeSession::prefill_stage_tokens(
    std::span<const std::int32_t> tokens
) {
    return impl_->session->prefill_stage_tokens(tokens);
}

StageResult NativeSession::prefill_stage_activations(
    std::span<const float> activations,
    std::size_t token_count
) {
    return impl_->session->prefill_stage_activations(activations, token_count);
}

StageResult NativeSession::decode_stage_token(std::int32_t token) {
    return impl_->session->decode_stage_token(token);
}

StageResult NativeSession::decode_stage_activation(
    std::span<const float> activation
) {
    return impl_->session->decode_stage_activation(activation);
}

void NativeSession::rollback(std::size_t token_count) {
    impl_->session->rollback(token_count);
}

class NativeEngine::Impl {
public:
    Impl(const std::string& package_path, EngineOptions options)
        : tensors(
              package_path,
              options.backend,
              options.threads,
              options.layer_start,
              options.layer_end
          ),
          architecture(create_architecture(tensors)) {
        prefill_attention_kernel = architecture->resolve_attention_kernel(
            options.backend, options.prefill_attention_kernel
        );
        decode_attention_kernel = architecture->resolve_attention_kernel(
            options.backend, options.decode_attention_kernel
        );
        info = architecture->model_info();
        info.source_sha256 = tensors.source_sha256();
        info.compute_backend = tensors.backend_name();
        info.compute_device = tensors.device_name();
        info.vocabulary_size = tensors.vocabulary_size();
        info.weight_bytes = tensors.weight_bytes();
        info.total_weight_bytes = tensors.total_weight_bytes();
        info.tensor_count = tensors.tensor_count();
        info.resident_layer_start = tensors.layer_start();
        info.resident_layer_end = tensors.layer_end();
    }

    TensorStore tensors;
    std::unique_ptr<ModelArchitecture> architecture;
    std::unique_ptr<InferenceSession> session;
    ModelInfo info;
    AttentionKernel prefill_attention_kernel = AttentionKernel::Unfused;
    AttentionKernel decode_attention_kernel = AttentionKernel::Unfused;

    InferenceSession& reset_session(std::size_t required_capacity) {
        if (!session || session->capacity() != required_capacity) {
            session = architecture->create_session(
                tensors, required_capacity,
                prefill_attention_kernel, decode_attention_kernel,
                LayerRange{tensors.layer_start(), tensors.layer_end()}
            );
        }
        session->reset();
        return *session;
    }
};

NativeEngine::NativeEngine(const std::string& package_path, Backend backend, int threads)
    : NativeEngine(package_path, EngineOptions{
          .backend = backend,
          .prefill_attention_kernel = AttentionKernel::Automatic,
          .decode_attention_kernel = AttentionKernel::Automatic,
          .threads = threads,
          .layer_start = 0,
          .layer_end = std::numeric_limits<std::uint32_t>::max(),
      }) {}

NativeEngine::NativeEngine(const std::string& package_path, EngineOptions options) {
    const auto start = std::chrono::steady_clock::now();
    impl_ = std::make_unique<Impl>(package_path, options);
    load_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
}

NativeEngine::~NativeEngine() = default;
NativeEngine::NativeEngine(NativeEngine&&) noexcept = default;
NativeEngine& NativeEngine::operator=(NativeEngine&&) noexcept = default;

const ModelInfo& NativeEngine::model_info() const { return impl_->info; }
AttentionKernel NativeEngine::prefill_attention_kernel() const {
    return impl_->prefill_attention_kernel;
}
AttentionKernel NativeEngine::decode_attention_kernel() const {
    return impl_->decode_attention_kernel;
}
double NativeEngine::load_time_ms() const { return load_time_ms_; }

std::unique_ptr<NativeSession> NativeEngine::create_session(std::size_t capacity) {
    if (capacity == 0 || capacity > impl_->info.context_length) {
        throw std::invalid_argument("native session capacity is outside model context");
    }
    auto session = impl_->architecture->create_session(
        impl_->tensors,
        capacity,
        impl_->prefill_attention_kernel,
        impl_->decode_attention_kernel,
        LayerRange{impl_->tensors.layer_start(), impl_->tensors.layer_end()}
    );
    return std::unique_ptr<NativeSession>(new NativeSession(
        std::make_unique<NativeSession::Impl>(std::move(session))
    ));
}

void NativeEngine::release_session() { impl_->session.reset(); }

std::vector<float> NativeEngine::logits(std::span<const std::int32_t> tokens) {
    if (tokens.empty() || tokens.size() > impl_->info.context_length) {
        throw std::invalid_argument("native token sequence is outside model context");
    }
    for (const std::int32_t token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= impl_->info.vocabulary_size) {
            throw std::invalid_argument("native token ID is outside model vocabulary");
        }
    }
    auto session = impl_->architecture->create_session(
        impl_->tensors, tokens.size(),
        impl_->prefill_attention_kernel, impl_->decode_attention_kernel,
        LayerRange{impl_->tensors.layer_start(), impl_->tensors.layer_end()}
    );
    return session->prefill_logits(tokens);
}

std::vector<std::vector<float>> NativeEngine::forced_decode_logits(
    std::span<const std::int32_t> prompt_tokens,
    std::span<const std::int32_t> forced_tokens
) {
    if (prompt_tokens.empty() ||
        prompt_tokens.size() + forced_tokens.size() > impl_->info.context_length) {
        throw std::invalid_argument("native logit trace is outside model context");
    }
    const auto validate_token = [this](std::int32_t token) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= impl_->info.vocabulary_size) {
            throw std::invalid_argument("native token ID is outside model vocabulary");
        }
    };
    for (const std::int32_t token : prompt_tokens) validate_token(token);
    for (const std::int32_t token : forced_tokens) validate_token(token);

    auto session = impl_->architecture->create_session(
        impl_->tensors, prompt_tokens.size() + forced_tokens.size(),
        impl_->prefill_attention_kernel, impl_->decode_attention_kernel,
        LayerRange{impl_->tensors.layer_start(), impl_->tensors.layer_end()}
    );
    std::vector<std::vector<float>> trace;
    trace.reserve(forced_tokens.size() + 1U);
    trace.push_back(session->prefill_logits(prompt_tokens));
    for (const std::int32_t token : forced_tokens) {
        trace.push_back(session->decode_logits(token));
    }
    return trace;
}

GenerationResult NativeEngine::generate(
    std::span<const std::int32_t> prompt_tokens,
    std::uint32_t max_tokens
) {
    if (prompt_tokens.empty() || max_tokens == 0) {
        throw std::invalid_argument("native generation requires prompt and output tokens");
    }
    const std::size_t required_capacity = prompt_tokens.size() + max_tokens - 1U;
    if (required_capacity > impl_->info.context_length) {
        throw std::invalid_argument("native generation exceeds model context");
    }
    for (const std::int32_t token : prompt_tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= impl_->info.vocabulary_size) {
            throw std::invalid_argument("native token ID is outside model vocabulary");
        }
    }
    InferenceSession& session = impl_->reset_session(required_capacity);
    GenerationResult result;
    double total_ms = 0.0;
    double decode_ms = 0.0;
    for (std::uint32_t index = 0; index < max_tokens; ++index) {
        const auto start = std::chrono::steady_clock::now();
        std::int32_t token = -1;
        if (index == 0) {
            PrefillResult prefill = session.prefill_greedy(prompt_tokens);
            token = prefill.token;
            result.prefill = prefill.metrics;
        } else {
            token = session.decode_greedy(result.sampled_tokens.back());
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();
        if (index == 0) result.time_to_first_token_ms = elapsed;
        else decode_ms += elapsed;
        total_ms += elapsed;
        result.sampled_tokens.push_back(token);
    }
    if (max_tokens > 1) {
        result.inter_token_latency_ms = decode_ms / static_cast<double>(max_tokens - 1);
        result.decode_tokens_per_second = 1000.0 / result.inter_token_latency_ms;
    }
    result.end_to_end_tokens_per_second =
        1000.0 * static_cast<double>(max_tokens) / total_ms;
    result.buffers = session.execution_buffers();
    return result;
}

const char * backend_name(Backend backend) {
    return backend == Backend::Cuda ? "cuda" : "cpu";
}

const char * attention_kernel_name(AttentionKernel kernel) {
    switch (kernel) {
    case AttentionKernel::Automatic: return "automatic";
    case AttentionKernel::Unfused: return "unfused";
    case AttentionKernel::Flash: return "flash";
    }
    return "unknown";
}

} // namespace jetsonfabric::native
