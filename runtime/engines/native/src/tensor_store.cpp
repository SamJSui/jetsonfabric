#include "tensor_store.hpp"

#include "ggml-cpu.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
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

std::string sha256_hex(const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) output << std::setw(2) << unsigned(byte);
    return output.str();
}

} // namespace

TensorStore::TensorStore(
    const std::string& package_path,
    Backend backend,
    int threads,
    std::uint32_t layer_start,
    std::uint32_t layer_end
) : backend_(create_backend(backend, threads)) {
    scheduler_backends_[scheduler_backend_count_++] = backend_.get();
    if (backend == Backend::Cuda) {
        cpu_fallback_ = create_backend(Backend::Cpu, threads);
        scheduler_backends_[scheduler_backend_count_++] = cpu_fallback_.get();
    }
    backend_name_ = ggml_backend_name(backend_.get());
    const ggml_backend_dev_t device = ggml_backend_get_device(backend_.get());
    device_name_ = device == nullptr ? "unknown" : ggml_backend_dev_name(device);
    jf_model * raw_model = nullptr;
    const jf_stage_plan plan{
        .layer_start = layer_start,
        .layer_end = layer_end,
        .verify_hashes = 1,
        .evict_before_open = 0,
    };
    require_ok(jf_model_open(package_path.c_str(), &plan, &raw_model), "open JFM package");
    Model model(raw_model);
    const void * metadata = nullptr;
    std::size_t metadata_size = 0;
    require_ok(
        jf_model_get_gguf_metadata(model.get(), &metadata, &metadata_size),
        "read JFM metadata"
    );
    metadata_.resize(metadata_size);
    std::memcpy(metadata_.data(), metadata, metadata_size);
    std::array<std::uint8_t, 32> source_sha{};
    require_ok(
        jf_model_get_source_sha256(model.get(), source_sha.data()),
        "read JFM source identity"
    );
    source_sha256_ = sha256_hex(source_sha);
    const jf_model_stats stats = jf_model_get_stats(model.get());
    layer_start_ = stats.layer_start;
    layer_end_ = stats.layer_end;
    weight_bytes_ = stats.selected_weight_bytes;
    total_weight_bytes_ = stats.total_weight_bytes;
    tensor_count_ = stats.selected_tensor_count;
    create_tensors(model.get());
    buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend_.get()));
    if (!buffer_) {
        throw std::runtime_error("could not allocate native model weights on backend");
    }
    if (host_context_) {
        host_buffer_.reset(
            ggml_backend_alloc_ctx_tensors(host_context_.get(), cpu_fallback_.get())
        );
        if (!host_buffer_) {
            throw std::runtime_error("could not allocate native host-resident weights");
        }
    }
    copy_tensors(model.get());
    ggml_backend_synchronize(backend_.get());
}

ggml_backend_ptr TensorStore::create_backend(Backend backend, int threads) {
    ggml_backend_load_all();
    const char * registry_name = backend == Backend::Cuda ? "CUDA" : "CPU";
    const ggml_backend_reg_t registry = ggml_backend_reg_by_name(registry_name);
    if (registry == nullptr || ggml_backend_reg_dev_count(registry) == 0) {
        throw std::runtime_error(
            std::string("native ") + registry_name + " backend is not available"
        );
    }
    const ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry, 0);
    ggml_backend_t raw_backend = ggml_backend_dev_init(device, nullptr);
    if (raw_backend == nullptr) {
        throw std::runtime_error("could not initialize native compute backend");
    }
    if (backend == Backend::Cpu) ggml_backend_cpu_set_n_threads(raw_backend, threads);
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
    if (cpu_fallback_) {
        host_context_.reset(ggml_init(ggml_init_params{
            .mem_size = metadata_bytes,
            .mem_buffer = nullptr,
            .no_alloc = true,
        }));
        if (!host_context_) {
            throw std::runtime_error("could not allocate native host tensor metadata");
        }
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
        const std::string name(view.name, view.name_length);
        ggml_context * tensor_context =
            host_context_ && name == "token_embd.weight"
            ? host_context_.get()
            : context_.get();
        ggml_tensor * tensor = ggml_new_tensor(
            tensor_context,
            to_ggml_type(view.type),
            static_cast<int>(view.rank),
            shape.data()
        );
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

ggml_tensor * TensorStore::find(const std::string& name) const {
    const auto found = tensors_.find(name);
    return found == tensors_.end() ? nullptr : found->second;
}

ggml_tensor * TensorStore::require(const std::string& name) const {
    ggml_tensor * tensor = find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("required native tensor is missing: " + name);
    }
    return tensor;
}

std::uint32_t TensorStore::vocabulary_size() const {
    ggml_tensor * vocabulary_tensor = find("token_embd.weight");
    if (vocabulary_tensor == nullptr) vocabulary_tensor = find("output.weight");
    if (vocabulary_tensor == nullptr) return 0;
    const std::int64_t size = vocabulary_tensor->ne[1];
    if (size <= 0 || size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("native vocabulary size is outside uint32 range");
    }
    return static_cast<std::uint32_t>(size);
}

} // namespace jetsonfabric::native
