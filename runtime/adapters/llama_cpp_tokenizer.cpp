#include "adapters/llama_cpp_tokenizer.hpp"

#include "gguf.h"
#include "llama-model-loader.h"
#include "llama-vocab.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

class GgufDeleter {
public:
    void operator()(gguf_context* context) const { gguf_free(context); }
};

using Gguf = std::unique_ptr<gguf_context, GgufDeleter>;

std::vector<char> read_metadata(const std::string& package_path) {
    const std::filesystem::path path =
        std::filesystem::path(package_path) / "metadata.gguf";
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("native model package is missing metadata.gguf");
    }
    std::vector<char> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    if (bytes.empty()) {
        throw std::invalid_argument("native model metadata is empty");
    }
    return bytes;
}

} // namespace

class LlamaCppTokenizer::Impl {
public:
    explicit Impl(const std::string& package_path) {
        const std::vector<char> bytes = read_metadata(package_path);
        Gguf metadata(gguf_init_from_buffer(
            bytes.data(),
            bytes.size(),
            gguf_init_params{.no_alloc = true, .ctx = nullptr}
        ));
        if (!metadata) {
            throw std::invalid_argument("native model metadata is not valid GGUF");
        }

        std::vector<std::string> splits;
        llama_model_loader loader(
            metadata.get(),
            nullptr,
            nullptr,
            "",
            splits,
            nullptr,
            false,
            false,
            false,
            true,
            nullptr,
            nullptr
        );
        vocab.load(loader, loader.llm_kv);
    }

    llama_vocab vocab;
};

LlamaCppTokenizer::LlamaCppTokenizer(const std::string& package_path)
    : impl_(std::make_unique<Impl>(package_path)) {}

LlamaCppTokenizer::~LlamaCppTokenizer() = default;
LlamaCppTokenizer::LlamaCppTokenizer(LlamaCppTokenizer&&) noexcept = default;
LlamaCppTokenizer& LlamaCppTokenizer::operator=(LlamaCppTokenizer&&) noexcept = default;

std::vector<std::int32_t> LlamaCppTokenizer::tokenize(std::string_view text) const {
    const std::vector<llama_token> tokens = impl_->vocab.tokenize(
        std::string(text),
        true,
        true
    );
    if (tokens.empty()) {
        throw std::invalid_argument("native prompt tokenization produced no tokens");
    }
    return {tokens.begin(), tokens.end()};
}

std::string LlamaCppTokenizer::token_piece(std::int32_t token) const {
    std::array<char, 256> small{};
    int32_t size = impl_->vocab.token_to_piece(
        static_cast<llama_token>(token),
        small.data(),
        static_cast<std::int32_t>(small.size()),
        0,
        true
    );
    if (size >= 0) {
        return {small.data(), static_cast<std::size_t>(size)};
    }
    std::string large(static_cast<std::size_t>(-size), '\0');
    size = impl_->vocab.token_to_piece(
        static_cast<llama_token>(token),
        large.data(),
        static_cast<std::int32_t>(large.size()),
        0,
        true
    );
    if (size < 0) throw std::runtime_error("failed to decode native token text");
    large.resize(static_cast<std::size_t>(size));
    return large;
}

bool LlamaCppTokenizer::is_end_token(std::int32_t token) const {
    return token == LLAMA_TOKEN_NULL ||
        impl_->vocab.is_eog(static_cast<llama_token>(token));
}

} // namespace jetsonfabric::runtime::adapters
