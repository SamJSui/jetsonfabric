#include "protocol/utf8.hpp"

#include <stdexcept>

namespace jetsonfabric::runtime::protocol {
namespace {

bool is_continuation(unsigned char value) {
    return (value & 0xc0U) == 0x80U;
}

std::size_t sequence_width(unsigned char lead) {
    if (lead <= 0x7fU) return 1;
    if (lead >= 0xc2U && lead <= 0xdfU) return 2;
    if (lead >= 0xe0U && lead <= 0xefU) return 3;
    if (lead >= 0xf0U && lead <= 0xf4U) return 4;
    throw std::invalid_argument("token text contains malformed UTF-8");
}

void validate_sequence(const std::string& value, std::size_t offset, std::size_t width) {
    const auto lead = static_cast<unsigned char>(value[offset]);
    for (std::size_t index = 1; index < width; ++index) {
        if (!is_continuation(static_cast<unsigned char>(value[offset + index]))) {
            throw std::invalid_argument("token text contains malformed UTF-8");
        }
    }

    const auto second = static_cast<unsigned char>(value[offset + 1]);
    if ((lead == 0xe0U && second < 0xa0U) ||
        (lead == 0xedU && second > 0x9fU) ||
        (lead == 0xf0U && second < 0x90U) ||
        (lead == 0xf4U && second > 0x8fU)) {
        throw std::invalid_argument("token text contains malformed UTF-8");
    }
}

} // namespace

std::size_t complete_utf8_prefix(const std::string& value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
        const std::size_t width = sequence_width(static_cast<unsigned char>(value[offset]));
        if (value.size() - offset < width) {
            return offset;
        }
        if (width > 1) {
            validate_sequence(value, offset, width);
        }
        offset += width;
    }
    return offset;
}

bool is_valid_utf8(const std::string& value) {
    try {
        return complete_utf8_prefix(value) == value.size();
    } catch (const std::invalid_argument&) {
        return false;
    }
}

} // namespace jetsonfabric::runtime::protocol
