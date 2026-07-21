#pragma once

#include <string_view>

namespace jetsonfabric::runtime::transport {

bool matches_media_type(std::string_view value, std::string_view expected);

} // namespace jetsonfabric::runtime::transport
