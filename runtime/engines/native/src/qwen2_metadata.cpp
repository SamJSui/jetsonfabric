#include "qwen2_metadata.hpp"

#include "gguf.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace jetsonfabric::native {
namespace {

class GgufDeleter {
public:
    void operator()(gguf_context * context) const { gguf_free(context); }
};

using Gguf = std::unique_ptr<gguf_context, GgufDeleter>;

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

Qwen2HParams load_qwen2_hparams(const std::filesystem::path& package_path) {
    const std::filesystem::path metadata_path = package_path / "metadata.gguf";
    Gguf context(gguf_init_from_file(
        metadata_path.string().c_str(),
        gguf_init_params{.no_alloc = true, .ctx = nullptr}
    ));
    if (!context) {
        throw std::runtime_error("could not parse preserved GGUF metadata");
    }
    const std::int64_t architecture_key = require_key(context.get(), "general.architecture");
    if (gguf_get_kv_type(context.get(), architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(context.get(), architecture_key)) != "qwen2") {
        throw std::runtime_error("native execution currently supports only qwen2 GGUF models");
    }

    Qwen2HParams params;
    ModelInfo& info = params.public_info;
    info.architecture = "qwen2";
    info.layer_count = require_u32(context.get(), "qwen2.block_count");
    info.embedding_length = require_u32(context.get(), "qwen2.embedding_length");
    params.feed_forward_length = require_u32(context.get(), "qwen2.feed_forward_length");
    params.head_count = require_u32(context.get(), "qwen2.attention.head_count");
    params.kv_head_count = require_u32(context.get(), "qwen2.attention.head_count_kv");
    info.context_length = require_u32(context.get(), "qwen2.context_length");
    params.rope_dimension_count = require_u32(context.get(), "qwen2.rope.dimension_count");
    params.rope_frequency_base = require_f32(context.get(), "qwen2.rope.freq_base");
    params.rms_epsilon = require_f32(
        context.get(),
        "qwen2.attention.layer_norm_rms_epsilon"
    );
    validate_hparams(params);
    return params;
}

} // namespace jetsonfabric::native
