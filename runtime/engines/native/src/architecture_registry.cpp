#include "model_architecture.hpp"

#include "gguf.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace jetsonfabric::native {

std::unique_ptr<ModelArchitecture> create_qwen2_architecture(
    const std::filesystem::path& package_path
);

namespace {

class GgufDeleter {
public:
    void operator()(gguf_context * context) const { gguf_free(context); }
};

using Gguf = std::unique_ptr<gguf_context, GgufDeleter>;
using Factory = std::function<std::unique_ptr<ModelArchitecture>(const std::filesystem::path&)>;

std::string read_architecture(const std::filesystem::path& package_path) {
    const std::filesystem::path metadata_path = package_path / "metadata.gguf";
    Gguf context(gguf_init_from_file(
        metadata_path.string().c_str(),
        gguf_init_params{.no_alloc = true, .ctx = nullptr}
    ));
    if (!context) throw std::runtime_error("could not parse preserved GGUF metadata");
    const std::int64_t key = gguf_find_key(context.get(), "general.architecture");
    if (key < 0 || gguf_get_kv_type(context.get(), key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF general.architecture metadata is required");
    }
    return gguf_get_val_str(context.get(), key);
}

const std::unordered_map<std::string, Factory>& factories() {
    static const std::unordered_map<std::string, Factory> registry = {
        {"qwen2", create_qwen2_architecture},
    };
    return registry;
}

} // namespace

std::unique_ptr<ModelArchitecture> create_architecture(const std::string& package_path) {
    const std::filesystem::path path(package_path);
    const std::string architecture = read_architecture(path);
    const auto found = factories().find(architecture);
    if (found == factories().end()) {
        throw std::runtime_error("unsupported native model architecture: " + architecture);
    }
    return found->second(path);
}

} // namespace jetsonfabric::native
