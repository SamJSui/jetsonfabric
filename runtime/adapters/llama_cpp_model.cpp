#include "adapters/llama_cpp_model.hpp"

#include "llama-ext.h"
#include "llama.h"

#include <array>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

std::once_flag backend_init_once;

void ensure_backend_loaded() {
    std::call_once(backend_init_once, [] {
        ggml_backend_load_all();
    });
}

std::string model_metadata_string(const llama_model* model, const char* key) {
    std::vector<char> buffer(128, '\0');
    int32_t size = llama_model_meta_val_str(model, key, buffer.data(), buffer.size());
    if (size < 0) {
        return {};
    }
    if (static_cast<std::size_t>(size + 1) > buffer.size()) {
        buffer.assign(static_cast<std::size_t>(size + 1), '\0');
        size = llama_model_meta_val_str(model, key, buffer.data(), buffer.size());
        if (size < 0) {
            return {};
        }
    }
    return std::string(buffer.data(), static_cast<std::size_t>(size));
}

} // namespace

struct LlamaCppModel::Impl {
    explicit Impl(LlamaCppModelConfig config_in)
        : config(std::move(config_in)) {
        if (config.model_path.empty()) {
            throw std::runtime_error("llama.cpp model path is required");
        }
        if (config.layer_start < 0 || config.layer_end < 0 ||
            (config.layer_end != 0 && config.layer_start >= config.layer_end)) {
            throw std::runtime_error("invalid llama.cpp model layer residency range");
        }
        ensure_backend_loaded();
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = config.n_gpu_layers;
        params.layer_start = static_cast<std::uint32_t>(config.layer_start);
        params.layer_end = static_cast<std::uint32_t>(config.layer_end);
        model = llama_model_load_from_file(config.model_path.c_str(), params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load llama.cpp model: " + config.model_path);
        }
        vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            throw std::runtime_error("failed to get llama.cpp vocabulary");
        }
        architecture = model_metadata_string(model, "general.architecture");
        if (architecture.empty()) {
            throw std::runtime_error("llama.cpp model is missing general.architecture metadata");
        }
    }

    ~Impl() {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

    LlamaCppModelConfig config;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    std::string architecture;
};

LlamaCppModel::LlamaCppModel(LlamaCppModelConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LlamaCppModel::~LlamaCppModel() = default;

llama_model* LlamaCppModel::raw_model() const {
    return impl_->model;
}

const llama_vocab* LlamaCppModel::raw_vocab() const {
    return impl_->vocab;
}

int LlamaCppModel::n_embd() const {
    return llama_model_n_embd(impl_->model);
}

int LlamaCppModel::n_layer() const {
    return llama_model_n_layer(impl_->model);
}

int LlamaCppModel::n_vocab() const {
    return llama_vocab_n_tokens(impl_->vocab);
}

const std::string& LlamaCppModel::architecture() const {
    return impl_->architecture;
}

int LlamaCppModel::loaded_layer_start() const {
    return static_cast<int>(llama_model_loaded_layer_start(impl_->model));
}

int LlamaCppModel::loaded_layer_end() const {
    return static_cast<int>(llama_model_loaded_layer_end(impl_->model));
}

std::uint64_t LlamaCppModel::resident_weight_bytes() const {
    return llama_model_resident_tensor_bytes(impl_->model);
}

std::uint64_t LlamaCppModel::total_weight_bytes() const {
    return llama_model_size(impl_->model);
}

std::uint64_t LlamaCppModel::resident_tensor_count() const {
    return llama_model_resident_tensor_count(impl_->model);
}

std::vector<std::int32_t> LlamaCppModel::tokenize(std::string_view text, bool add_special) const {
    const std::string normalized = text.empty() ? "Hello" : std::string(text);
    int32_t count = llama_tokenize(
        impl_->vocab,
        normalized.data(),
        static_cast<int32_t>(normalized.size()),
        nullptr,
        0,
        add_special,
        true
    );
    if (count < 0) {
        count = -count;
    }
    if (count <= 0) {
        throw std::runtime_error("failed to count llama.cpp tokens");
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    const int32_t actual = llama_tokenize(
        impl_->vocab,
        normalized.data(),
        static_cast<int32_t>(normalized.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        true
    );
    if (actual < 0) {
        throw std::runtime_error("failed to tokenize llama.cpp prompt");
    }
    tokens.resize(static_cast<std::size_t>(actual));
    return std::vector<std::int32_t>(tokens.begin(), tokens.end());
}

std::string LlamaCppModel::token_piece(std::int32_t token) const {
    std::array<char, 256> small{};
    int32_t size = llama_token_to_piece(
        impl_->vocab,
        static_cast<llama_token>(token),
        small.data(),
        static_cast<int32_t>(small.size()),
        0,
        true
    );
    if (size >= 0) {
        return std::string(small.data(), static_cast<std::size_t>(size));
    }
    std::string large(static_cast<std::size_t>(-size), '\0');
    size = llama_token_to_piece(
        impl_->vocab,
        static_cast<llama_token>(token),
        large.data(),
        static_cast<int32_t>(large.size()),
        0,
        true
    );
    if (size < 0) {
        throw std::runtime_error("failed to convert llama.cpp token to text");
    }
    large.resize(static_cast<std::size_t>(size));
    return large;
}

bool LlamaCppModel::is_end_token(std::int32_t token) const {
    return token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(impl_->vocab, static_cast<llama_token>(token));
}

} // namespace jetsonfabric::runtime::adapters
