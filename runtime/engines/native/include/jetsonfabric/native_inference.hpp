#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jetsonfabric::native {

enum class Backend {
    Cpu,
    Cuda,
};

enum class AttentionKernel {
    Automatic,
    Unfused,
    Flash,
};

struct EngineOptions {
    Backend backend = Backend::Cpu;
    AttentionKernel prefill_attention_kernel = AttentionKernel::Automatic;
    AttentionKernel decode_attention_kernel = AttentionKernel::Automatic;
    int threads = 1;
    std::uint32_t layer_start = 0;
    std::uint32_t layer_end = std::numeric_limits<std::uint32_t>::max();
};

struct ModelInfo {
    std::string architecture;
    std::string source_sha256;
    std::string compute_backend;
    std::string compute_device;
    std::uint32_t layer_count = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t context_length = 0;
    std::uint32_t vocabulary_size = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t total_weight_bytes = 0;
    std::uint64_t tensor_count = 0;
    std::uint32_t resident_layer_start = 0;
    std::uint32_t resident_layer_end = 0;
};

struct StageMemoryEstimate {
    std::uint64_t resident_weight_bytes = 0;
    std::uint64_t reserved_kv_bytes = 0;
};

struct ActivationView {
    std::span<const std::uint8_t> bytes;
    std::size_t token_count = 0;

    bool empty() const noexcept { return bytes.empty(); }
};

struct ActivationBuffer {
    std::vector<std::uint8_t> bytes;
    std::size_t token_count = 0;

    bool empty() const noexcept { return bytes.empty(); }
    ActivationView view() const noexcept { return ActivationView{bytes, token_count}; }
};

struct StageResult {
    ActivationBuffer activation;
    std::int32_t sampled_token = -1;

    bool has_activations() const noexcept { return !activation.empty(); }
    bool has_sampled_token() const noexcept { return sampled_token >= 0; }
};

struct PrefillMetrics {
    bool plan_reused = false;
    bool attention_backend_verified = false;
    double planning_ms = 0.0;
    double allocation_ms = 0.0;
    double host_input_preparation_ms = 0.0;
    double compute_ms = 0.0;
    double output_read_ms = 0.0;
};

struct ExecutionBufferMetrics {
    std::uint64_t kv_cache_bytes = 0;
    std::uint64_t prefill_scratch_bytes = 0;
    std::uint64_t decode_scratch_bytes = 0;
    std::uint64_t prefill_host_input_bytes = 0;
};

struct GenerationResult {
    std::vector<std::int32_t> sampled_tokens;
    PrefillMetrics prefill;
    ExecutionBufferMetrics buffers;
    double time_to_first_token_ms = 0.0;
    double inter_token_latency_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    double end_to_end_tokens_per_second = 0.0;
};

class NativeSession {
public:
    ~NativeSession();

    NativeSession(NativeSession&&) noexcept;
    NativeSession& operator=(NativeSession&&) noexcept;

    NativeSession(const NativeSession&) = delete;
    NativeSession& operator=(const NativeSession&) = delete;

    std::size_t capacity() const;
    std::size_t position() const;
    std::int32_t prefill_greedy(std::span<const std::int32_t> tokens);
    std::int32_t decode_greedy(std::int32_t token);
    StageResult prefill_stage_tokens(std::span<const std::int32_t> tokens);
    StageResult prefill_stage_activations(ActivationView activation);
    StageResult decode_stage_token(std::int32_t token);
    StageResult decode_stage_activation(ActivationView activation);
    void reserve_execution_buffers();
    ExecutionBufferMetrics execution_buffers() const;
    void reset();
    void rollback(std::size_t token_count);

private:
    class Impl;
    explicit NativeSession(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
    friend class NativeEngine;
};

class NativeEngine {
public:
    NativeEngine(const std::string& package_path, Backend backend, int threads);
    NativeEngine(const std::string& package_path, EngineOptions options);
    ~NativeEngine();

    NativeEngine(NativeEngine&&) noexcept;
    NativeEngine& operator=(NativeEngine&&) noexcept;

    NativeEngine(const NativeEngine&) = delete;
    NativeEngine& operator=(const NativeEngine&) = delete;

    const ModelInfo& model_info() const;
    AttentionKernel prefill_attention_kernel() const;
    AttentionKernel decode_attention_kernel() const;
    double load_time_ms() const;

    std::unique_ptr<NativeSession> create_session(std::size_t capacity);

    std::vector<float> logits(std::span<const std::int32_t> tokens);
    std::vector<std::vector<float>> forced_decode_logits(
        std::span<const std::int32_t> prompt_tokens,
        std::span<const std::int32_t> forced_tokens
    );
    GenerationResult generate(
        std::span<const std::int32_t> prompt_tokens,
        std::uint32_t max_tokens
    );
    void release_session();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    double load_time_ms_ = 0.0;
};

const char * backend_name(Backend backend);
const char * attention_kernel_name(AttentionKernel kernel);

StageMemoryEstimate estimate_stage_memory(
    const std::string& package_path,
    Backend backend,
    std::uint32_t layer_start,
    std::uint32_t layer_end,
    std::size_t session_capacity,
    std::size_t session_count
);

} // namespace jetsonfabric::native
