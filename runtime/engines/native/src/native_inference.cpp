#include "jetsonfabric/native_inference.hpp"

#include "model_architecture.hpp"
#include "tensor_store.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::native {
class NativeEngine::Impl {
public:
    Impl(const std::string& package_path, Backend backend, int threads)
        : tensors(package_path, backend, threads),
          architecture(create_architecture(tensors)) {
        info = architecture->model_info();
        info.source_sha256 = tensors.source_sha256();
        info.compute_backend = tensors.backend_name();
        info.compute_device = tensors.device_name();
        info.vocabulary_size = tensors.vocabulary_size();
        info.weight_bytes = tensors.weight_bytes();
    }

    TensorStore tensors;
    std::unique_ptr<ModelArchitecture> architecture;
    std::unique_ptr<InferenceSession> session;
    ModelInfo info;

    InferenceSession& reset_session(std::size_t required_capacity) {
        if (!session || session->capacity() != required_capacity) {
            session = architecture->create_session(tensors, required_capacity);
        }
        session->reset();
        return *session;
    }
};

NativeEngine::NativeEngine(const std::string& package_path, Backend backend, int threads) {
    const auto start = std::chrono::steady_clock::now();
    impl_ = std::make_unique<Impl>(package_path, backend, threads);
    load_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
}

NativeEngine::~NativeEngine() = default;
NativeEngine::NativeEngine(NativeEngine&&) noexcept = default;
NativeEngine& NativeEngine::operator=(NativeEngine&&) noexcept = default;

const ModelInfo& NativeEngine::model_info() const { return impl_->info; }
double NativeEngine::load_time_ms() const { return load_time_ms_; }

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
    auto session = impl_->architecture->create_session(impl_->tensors, tokens.size());
    return session->prefill_logits(tokens);
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

} // namespace jetsonfabric::native
