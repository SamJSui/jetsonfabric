#include "jetsonfabric/native_inference.hpp"

#include "jetsonfabric/engine.h"
#include "qwen2_metadata.hpp"

#include "gguf.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace jetsonfabric::native {
namespace {

class ModelDeleter {
public:
    void operator()(jf_model * model) const { jf_model_close(model); }
};

class GgufDeleter {
public:
    void operator()(gguf_context * context) const { gguf_free(context); }
};

using Model = std::unique_ptr<jf_model, ModelDeleter>;
using Gguf = std::unique_ptr<gguf_context, GgufDeleter>;

void require_ok(jf_status status, const char * operation) {
    if (status.code == JF_STATUS_OK) return;
    const std::string message = std::string(operation) + ": " + status.message;
    switch (status.code) {
    case JF_STATUS_INVALID_ARGUMENT:
    case JF_STATUS_FORMAT_ERROR:
    case JF_STATUS_NOT_FOUND:
        throw std::invalid_argument(message);
    case JF_STATUS_OUT_OF_MEMORY:
        throw std::bad_alloc();
    case JF_STATUS_IO_ERROR:
    case JF_STATUS_OK:
        throw std::runtime_error(message);
    }
    throw std::runtime_error(message);
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::invalid_argument("native stage memory estimate overflows uint64");
    }
    return left * right;
}

std::size_t cache_capacity(
    std::size_t requested,
    const Qwen2HParams& params,
    Backend backend
) {
    if (backend != Backend::Cuda || !qwen2_supports_flash_attention(params)) {
        return requested;
    }
    constexpr std::size_t alignment = 256;
    const std::size_t aligned =
        (requested + alignment - 1U) / alignment * alignment;
    return std::min(aligned, static_cast<std::size_t>(params.public_info.context_length));
}

Gguf read_metadata(const jf_model * model) {
    const void * data = nullptr;
    std::size_t size = 0;
    require_ok(jf_model_get_gguf_metadata(model, &data, &size), "read JFM metadata");
    Gguf metadata(gguf_init_from_buffer(
        data,
        size,
        gguf_init_params{.no_alloc = true, .ctx = nullptr}
    ));
    if (!metadata) {
        throw std::invalid_argument("could not parse preserved GGUF metadata");
    }
    return metadata;
}

std::uint64_t estimate_kv_bytes(
    const Qwen2HParams& params,
    std::uint32_t resident_layers,
    std::size_t capacity,
    std::size_t session_count
) {
    const std::uint64_t head_length =
        params.public_info.embedding_length / params.head_count;
    const std::uint64_t kv_length = head_length * params.kv_head_count;
    std::uint64_t bytes = checked_multiply(2, resident_layers);
    bytes = checked_multiply(bytes, kv_length);
    bytes = checked_multiply(bytes, capacity);
    bytes = checked_multiply(bytes, sizeof(std::uint16_t));
    return checked_multiply(bytes, session_count);
}

} // namespace

StageMemoryEstimate estimate_stage_memory(
    const std::string& package_path,
    Backend backend,
    std::uint32_t layer_start,
    std::uint32_t layer_end,
    std::size_t session_capacity,
    std::size_t session_count
) {
    if (package_path.empty() || layer_start >= layer_end || session_capacity == 0 ||
        session_count == 0) {
        throw std::invalid_argument(
            "native stage memory estimate requires a package, layers, and sessions"
        );
    }

    jf_model * raw_model = nullptr;
    const jf_stage_plan plan{
        .layer_start = layer_start,
        .layer_end = layer_end,
        .verify_hashes = 0,
        .evict_before_open = 0,
    };
    require_ok(jf_model_open(package_path.c_str(), &plan, &raw_model), "inspect JFM package");
    const Model model(raw_model);
    const Gguf metadata = read_metadata(model.get());
    Qwen2HParams params;
    try {
        params = load_qwen2_hparams(metadata.get());
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(
            std::string("inspect native model metadata: ") + error.what()
        );
    }
    if (layer_end > params.public_info.layer_count ||
        session_capacity > params.public_info.context_length) {
        throw std::invalid_argument("native stage memory estimate exceeds model dimensions");
    }

    const jf_model_stats stats = jf_model_get_stats(model.get());
    if (stats.selected_weight_bytes == 0) {
        throw std::invalid_argument("native JFM stage contains no resident weights");
    }
    return StageMemoryEstimate{
        .resident_weight_bytes = stats.selected_weight_bytes,
        .reserved_kv_bytes = estimate_kv_bytes(
            params,
            layer_end - layer_start,
            cache_capacity(session_capacity, params, backend),
            session_count
        ),
    };
}

} // namespace jetsonfabric::native
