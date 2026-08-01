#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jetsonfabric::native {

enum class Backend {
    Cpu,
    Cuda,
};

struct ModelInfo {
    std::string architecture;
    std::uint32_t layer_count = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t context_length = 0;
    std::uint32_t vocabulary_size = 0;
    std::uint64_t weight_bytes = 0;
};

struct GenerationResult {
    std::vector<std::int32_t> sampled_tokens;
    double time_to_first_token_ms = 0.0;
    double inter_token_latency_ms = 0.0;
    double output_tokens_per_second = 0.0;
};

class NativeEngine {
public:
    NativeEngine(const std::string& package_path, Backend backend, int threads);
    ~NativeEngine();

    NativeEngine(NativeEngine&&) noexcept;
    NativeEngine& operator=(NativeEngine&&) noexcept;

    NativeEngine(const NativeEngine&) = delete;
    NativeEngine& operator=(const NativeEngine&) = delete;

    const ModelInfo& model_info() const;
    double load_time_ms() const;

    std::vector<float> logits(std::span<const std::int32_t> tokens);
    GenerationResult generate(
        std::span<const std::int32_t> prompt_tokens,
        std::uint32_t max_tokens
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    double load_time_ms_ = 0.0;
};

const char * backend_name(Backend backend);

} // namespace jetsonfabric::native
