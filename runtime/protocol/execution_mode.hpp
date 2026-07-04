#pragma once

#include <string>

namespace jetsonfabric::runtime {

enum class ExecutionMode {
    DataParallel,
    PipelineParallel,
    TensorParallel,
};

ExecutionMode parse_execution_mode(const std::string& value);
std::string execution_mode_string(ExecutionMode mode);
bool is_valid_execution_mode(const std::string& value);

} // namespace jetsonfabric::runtime