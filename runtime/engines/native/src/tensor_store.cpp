#include "tensor_store.hpp"

#include "ggml-cpu.h"
#ifdef JF_NATIVE_CUDA
#include "ggml-cuda.h"
#endif

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::native {
namespace {

class ModelDeleter {
public:
    void operator()(jf_model * model) const { jf_model_close(model); }
};

using Model = std::unique_ptr<jf_model, ModelDeleter>;

ggml_type to_ggml_type(std::uint32_t type) {
    static constexpr std::array<ggml_type, 36> types = {
        GGML_TYPE_COUNT, GGML_TYPE_F32, GGML_TYPE_F16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_I8, GGML_TYPE_I16,
        GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64, GGML_TYPE_IQ1_M,
        GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_MXFP4,
        GGML_TYPE_NVFP4, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0,
    };
    if (type == 0 || type >= types.size()) {
        throw std::runtime_error("unsupported JFM tensor type " + std::to_string(type));
    }
    return types[type];
}

void require_ok(jf_status status, const char * operation) {
    if (status.code != JF_STATUS_OK) {
        throw std::runtime_error(std::string(operation) + ": " + status.message);
    }
}

} // namespace

TensorStore::TensorStore(
    const std::string& package_path,
    std::uint32_t layer_count,
    Backend backend,
    int threads
) : backend_(create_backend(backend, threads)) {
    jf_model * raw_model = nullptr;
    const jf_stage_plan plan{
        .layer_start = 0,
        .layer_end = layer_count,
        .verify_hashes = 0,
        .evict_before_open = 0,
    };
    require_ok(jf_model_open(package_path.c_str(), &plan, &raw_model), "open JFM package");
    Model model(raw_model);
    weight_bytes_ = jf_model_get_stats(model.get()).selected_weight_bytes;
    create_tensors(model.get());
    buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend_.get()));
    if (!buffer_) {
        throw std::runtime_error("could not allocate native model weights on backend");
    }
    copy_tensors(model.get());
    ggml_backend_synchronize(backend_.get());
}

ggml_backend_ptr TensorStore::create_backend(Backend backend, int threads) {
    ggml_backend_t raw_backend = nullptr;
    if (backend == Backend::Cuda) {
#ifdef JF_NATIVE_CUDA
        raw_backend = ggml_backend_cuda_init(0);
#else
        throw std::runtime_error("native CUDA backend was not compiled");
#endif
    } else {
        raw_backend = ggml_backend_cpu_init();
        if (raw_backend != nullptr) {
            ggml_backend_cpu_set_n_threads(raw_backend, threads);
        }
    }
    if (raw_backend == nullptr) {
        throw std::runtime_error("could not initialize native compute backend");
    }
    return ggml_backend_ptr(raw_backend);
}

void TensorStore::create_tensors(jf_model * model) {
    const std::size_t tensor_count = jf_model_tensor_count(model);
    const std::size_t metadata_bytes = tensor_count * ggml_tensor_overhead() + 1024U * 1024U;
    context_.reset(ggml_init(ggml_init_params{
        .mem_size = metadata_bytes,
        .mem_buffer = nullptr,
        .no_alloc = true,
    }));
    if (!context_) {
        throw std::runtime_error("could not allocate native tensor metadata");
    }
    for (std::size_t index = 0; index < tensor_count; ++index) {
        jf_tensor_view view{};
        require_ok(jf_model_tensor_at(model, index, &view), "read JFM tensor");
        std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
        for (std::uint32_t dimension = 0; dimension < view.rank; ++dimension) {
            if (view.shape[dimension] > static_cast<std::uint64_t>(INT64_MAX)) {
                throw std::runtime_error("tensor dimension exceeds GGML range");
            }
            shape[dimension] = static_cast<std::int64_t>(view.shape[dimension]);
        }
        ggml_tensor * tensor = ggml_new_tensor(
            context_.get(),
            to_ggml_type(view.type),
            static_cast<int>(view.rank),
            shape.data()
        );
        const std::string name(view.name, view.name_length);
        ggml_set_name(tensor, name.c_str());
        if (!tensors_.emplace(name, tensor).second) {
            throw std::runtime_error("duplicate native tensor " + name);
        }
    }
}

void TensorStore::copy_tensors(jf_model * model) {
    for (std::size_t index = 0; index < jf_model_tensor_count(model); ++index) {
        jf_tensor_view view{};
        require_ok(jf_model_tensor_at(model, index, &view), "read JFM tensor");
        const std::string name(view.name, view.name_length);
        ggml_tensor * tensor = require(name);
        if (ggml_nbytes(tensor) != view.size) {
            throw std::runtime_error("GGML byte size disagrees with JFM tensor " + name);
        }
        ggml_backend_tensor_set(tensor, view.data, 0, static_cast<std::size_t>(view.size));
    }
}

ggml_tensor * TensorStore::require(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error("required Qwen2 tensor is missing: " + name);
    }
    return found->second;
}

std::uint32_t TensorStore::vocabulary_size() const {
    const std::int64_t size = require("token_embd.weight")->ne[1];
    if (size <= 0 || size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Qwen2 vocabulary size is outside uint32 range");
    }
    return static_cast<std::uint32_t>(size);
}

} // namespace jetsonfabric::native
