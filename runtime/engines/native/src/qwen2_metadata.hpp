#pragma once

#include "jetsonfabric/native_inference.hpp"

#include <cstdint>
struct gguf_context;

namespace jetsonfabric::native {

struct Qwen2HParams {
    ModelInfo public_info;
    std::uint32_t feed_forward_length = 0;
    std::uint32_t head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t rope_dimension_count = 0;
    float rope_frequency_base = 0.0F;
    float rms_epsilon = 0.0F;
};

Qwen2HParams load_qwen2_hparams(const gguf_context * metadata);
bool qwen2_supports_flash_attention(const Qwen2HParams& params) noexcept;

} // namespace jetsonfabric::native
