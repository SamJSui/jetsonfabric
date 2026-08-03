#pragma once

#include "tokenization/tokenizer.hpp"

#include <memory>
#include <string>

namespace jetsonfabric::runtime::adapters {

// Loads only preserved GGUF metadata from a JFM package. This keeps tokenizer
// parity during native serving without loading a second copy of model weights.
class LlamaCppTokenizer final : public tokenization::Tokenizer {
public:
    explicit LlamaCppTokenizer(const std::string& package_path);
    ~LlamaCppTokenizer() override;

    LlamaCppTokenizer(LlamaCppTokenizer&&) noexcept;
    LlamaCppTokenizer& operator=(LlamaCppTokenizer&&) noexcept;

    LlamaCppTokenizer(const LlamaCppTokenizer&) = delete;
    LlamaCppTokenizer& operator=(const LlamaCppTokenizer&) = delete;

    std::vector<std::int32_t> tokenize(std::string_view text) const override;
    std::string token_piece(std::int32_t token) const override;
    bool is_end_token(std::int32_t token) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
