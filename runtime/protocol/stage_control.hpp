#pragma once

#include <stdexcept>
#include <string>

namespace jetsonfabric::runtime::protocol {

inline constexpr const char* kStageOperationExecute = "execute";
inline constexpr const char* kStageOperationCloseSession = "close_session";
inline constexpr const char* kStageOperationRollbackSession = "rollback_session";

std::string decode_stage_operation(const std::string& frame);

inline void validate_stage_operation_name(const std::string& operation) {
    if (operation != kStageOperationExecute &&
        operation != kStageOperationCloseSession &&
        operation != kStageOperationRollbackSession) {
        throw std::invalid_argument("invalid stage operation: " + operation);
    }
}

inline void validate_stage_operation(const std::string& operation, int rollback_tokens) {
    validate_stage_operation_name(operation);
    if (operation == kStageOperationRollbackSession && rollback_tokens <= 0) {
        throw std::invalid_argument("rollback_session requires positive rollback_tokens");
    }
    if (operation != kStageOperationRollbackSession && rollback_tokens != 0) {
        throw std::invalid_argument("rollback_tokens is only valid for rollback_session");
    }
}

} // namespace jetsonfabric::runtime::protocol
