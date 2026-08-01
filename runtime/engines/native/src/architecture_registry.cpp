#include "model_architecture.hpp"

#include "tensor_store.hpp"

#include "gguf.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace jetsonfabric::native {

std::unique_ptr<ModelArchitecture> create_qwen2_architecture(
    const gguf_context * metadata,
    const TensorStore& tensors
);

namespace {

class GgufDeleter {
public:
    void operator()(gguf_context * context) const { gguf_free(context); }
};

using Gguf = std::unique_ptr<gguf_context, GgufDeleter>;
using Factory = std::function<std::unique_ptr<ModelArchitecture>(
    const gguf_context *, const TensorStore&
)>;

Gguf parse_metadata(const TensorStore& tensors) {
    const auto metadata = tensors.gguf_metadata();
    Gguf context(gguf_init_from_buffer(
        metadata.data(), metadata.size(),
        gguf_init_params{.no_alloc = true, .ctx = nullptr}
    ));
    if (!context) throw std::runtime_error("could not parse preserved GGUF metadata");
    return context;
}

std::string read_architecture(const gguf_context * context) {
    const std::int64_t key = gguf_find_key(context, "general.architecture");
    if (key < 0 || gguf_get_kv_type(context, key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF general.architecture metadata is required");
    }
    return gguf_get_val_str(context, key);
}

const std::unordered_map<std::string, Factory>& factories() {
    static const std::unordered_map<std::string, Factory> registry = {
        {"qwen2", create_qwen2_architecture},
    };
    return registry;
}

} // namespace

std::unique_ptr<ModelArchitecture> create_architecture(const TensorStore& tensors) {
    const Gguf metadata = parse_metadata(tensors);
    const std::string architecture = read_architecture(metadata.get());
    const auto found = factories().find(architecture);
    if (found == factories().end()) {
        throw std::runtime_error("unsupported native model architecture: " + architecture);
    }
    return found->second(metadata.get(), tensors);
}

} // namespace jetsonfabric::native
