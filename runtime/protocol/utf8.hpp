#pragma once

#include <cstddef>
#include <string>

namespace jetsonfabric::runtime::protocol {

// Returns the number of bytes that form complete UTF-8 code points. An
// incomplete trailing code point is excluded; malformed UTF-8 throws.
std::size_t complete_utf8_prefix(const std::string& value);

bool is_valid_utf8(const std::string& value);

} // namespace jetsonfabric::runtime::protocol
