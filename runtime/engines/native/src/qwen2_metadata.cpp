#include "qwen2_metadata.hpp"

#include "gguf.h"

#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace jetsonfabric::native {
namespace {

std::int64_t require_key(const gguf_context * context, const char * key) {
    const std::int64_t index = gguf_find_key(context, key);
    if (index < 0) {
        throw std::runtime_error(std::string("missing GGUF metadata key ") + key);
    }
    return index;
}

std::uint32_t require_u32(const gguf_context * context, const char * key) {
    const std::int64_t index = require_key(context, key);
    if (gguf_get_kv_type(context, index) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("GGUF metadata key is not uint32: ") + key);
    }
    return gguf_get_val_u32(context, index);
}

float require_f32(const gguf_context * context, const char * key) {
    const std::int64_t index = require_key(context, key);
    if (gguf_get_kv_type(context, index) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("GGUF metadata key is not float32: ") + key);
    }
    return gguf_get_val_f32(context, index);
}

void reject_unsupported_rope_scaling(const gguf_context * context) {
    const std::int64_t type_key = gguf_find_key(context, "qwen2.rope.scaling.type");
    if (type_key >= 0) {
        if (gguf_get_kv_type(context, type_key) != GGUF_TYPE_STRING) {
            throw std::runtime_error("Qwen2 RoPE scaling type is not a string");
        }
        const std::string type = gguf_get_val_str(context, type_key);
        if (type != "none" && type != "linear") {
            throw std::runtime_error("native Qwen2 does not support RoPE scaling type " + type);
        }
    }
    const std::int64_t factor_key = gguf_find_key(context, "qwen2.rope.scaling.factor");
    if (factor_key >= 0) {
        if (gguf_get_kv_type(context, factor_key) != GGUF_TYPE_FLOAT32 ||
            std::fabs(gguf_get_val_f32(context, factor_key) - 1.0F) > 1.0e-6F) {
            throw std::runtime_error("native Qwen2 does not support scaled RoPE factors");
        }
    }
}

void validate_hparams(const Qwen2HParams& params) {
    const ModelInfo& info = params.public_info;
    if (info.layer_count == 0 || info.embedding_length == 0 ||
        params.feed_forward_length == 0 || params.head_count == 0 ||
        params.kv_head_count == 0 || info.context_length == 0) {
        throw std::runtime_error("Qwen2 metadata dimensions must be positive");
    }
    if (info.embedding_length % params.head_count != 0 ||
        params.head_count % params.kv_head_count != 0) {
        throw std::runtime_error("Qwen2 attention dimensions are inconsistent");
    }
    const std::uint32_t head_length = info.embedding_length / params.head_count;
    if (params.rope_dimension_count != head_length || params.rms_epsilon <= 0.0F ||
        params.rope_frequency_base <= 0.0F) {
        throw std::runtime_error("Qwen2 RoPE or RMS metadata is unsupported");
    }
}

} // namespace

Qwen2HParams load_qwen2_hparams(const gguf_context * context) {
    if (context == nullptr) throw std::invalid_argument("Qwen2 metadata is required");
    const std::int64_t architecture_key = require_key(context, "general.architecture");
    if (gguf_get_kv_type(context, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(context, architecture_key)) != "qwen2") {
        throw std::runtime_error("native execution currently supports only qwen2 GGUF models");
    }
    reject_unsupported_rope_scaling(context);

    Qwen2HParams params;
    ModelInfo& info = params.public_info;
    info.architecture = "qwen2";
    info.layer_count = require_u32(context, "qwen2.block_count");
    info.embedding_length = require_u32(context, "qwen2.embedding_length");
    params.feed_forward_length = require_u32(context, "qwen2.feed_forward_length");
    params.head_count = require_u32(context, "qwen2.attention.head_count");
    params.kv_head_count = require_u32(context, "qwen2.attention.head_count_kv");
    info.context_length = require_u32(context, "qwen2.context_length");
    params.rope_dimension_count = require_u32(context, "qwen2.rope.dimension_count");
    params.rope_frequency_base = require_f32(context, "qwen2.rope.freq_base");
    params.rms_epsilon = require_f32(
        context,
        "qwen2.attention.layer_norm_rms_epsilon"
    );
    validate_hparams(params);
    return params;
}

} // namespace jetsonfabric::native
