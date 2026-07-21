#include "transport/media_type.hpp"

#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace jetsonfabric::runtime::transport {
namespace {

bool is_token_character(unsigned char character) {
    return std::isalnum(character) ||
           std::string_view("!#$%&'*+-.^_`|~").find(static_cast<char>(character)) != std::string_view::npos;
}

class MediaTypeParser {
public:
    explicit MediaTypeParser(std::string_view value) : value_(value) {}

    std::optional<std::string> parse() {
        skip_whitespace();
        const std::string type = read_token();
        if (type.empty() || !consume('/')) return std::nullopt;
        const std::string subtype = read_token();
        if (subtype.empty()) return std::nullopt;

        while (true) {
            skip_whitespace();
            if (position_ == value_.size()) return type + "/" + subtype;
            if (!consume(';')) return std::nullopt;
            skip_whitespace();
            if (read_token().empty()) return std::nullopt;
            skip_whitespace();
            if (!consume('=')) return std::nullopt;
            skip_whitespace();
            if (!read_parameter_value()) return std::nullopt;
        }
    }

private:
    void skip_whitespace() {
        while (position_ < value_.size() &&
               (value_[position_] == ' ' || value_[position_] == '\t')) {
            ++position_;
        }
    }

    bool consume(char expected) {
        if (position_ >= value_.size() || value_[position_] != expected) return false;
        ++position_;
        return true;
    }

    std::string read_token() {
        std::string token;
        while (position_ < value_.size() &&
               is_token_character(static_cast<unsigned char>(value_[position_]))) {
            token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(value_[position_]))));
            ++position_;
        }
        return token;
    }

    bool read_parameter_value() {
        if (position_ >= value_.size()) return false;
        if (value_[position_] != '"') return !read_token().empty();

        ++position_;
        while (position_ < value_.size()) {
            const unsigned char character = static_cast<unsigned char>(value_[position_++]);
            if (character == '"') return true;
            if (character == '\\') {
                if (position_ >= value_.size()) return false;
                const unsigned char escaped = static_cast<unsigned char>(value_[position_++]);
                if (escaped == '\r' || escaped == '\n' || escaped == 0x7fU) return false;
                continue;
            }
            if ((character < 0x20U && character != '\t') || character == 0x7fU) return false;
        }
        return false;
    }

    std::string_view value_;
    std::size_t position_ = 0;
};

std::optional<std::string> parse_media_type(std::string_view value) {
    return MediaTypeParser(value).parse();
}

} // namespace

bool matches_media_type(std::string_view value, std::string_view expected) {
    const std::optional<std::string> actual_type = parse_media_type(value);
    const std::optional<std::string> expected_type = parse_media_type(expected);
    return actual_type.has_value() && expected_type.has_value() && *actual_type == *expected_type;
}

} // namespace jetsonfabric::runtime::transport
