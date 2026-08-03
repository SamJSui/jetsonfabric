#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace jetsonfabric::runtime::tokenization {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<std::int32_t> tokenize(std::string_view text) const = 0;
    virtual std::string token_piece(std::int32_t token) const = 0;
    virtual bool is_end_token(std::int32_t token) const = 0;
};

} // namespace jetsonfabric::runtime::tokenization
