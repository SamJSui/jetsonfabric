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

} // namespace jetsonfabric::runtime
