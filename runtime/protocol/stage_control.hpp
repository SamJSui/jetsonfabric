#pragma once

#include <string>

namespace jetsonfabric::runtime::protocol {

inline constexpr const char* kStageOperationExecute = "execute";
inline constexpr const char* kStageOperationCloseSession = "close_session";

std::string decode_stage_operation(const std::string& frame);

} // namespace jetsonfabric::runtime::protocol
