#include "protocol/execution_mode.hpp"

#include <stdexcept>

namespace jetsonfabric::runtime {

ExecutionMode parse_execution_mode(const std::string& value) {
    if (value == "data_parallel") {
        return ExecutionMode::DataParallel;
    }
    if (value == "pipeline_parallel") {
        return ExecutionMode::PipelineParallel;
    }
    if (value == "tensor_parallel") {
        return ExecutionMode::TensorParallel;
    }

    throw std::invalid_argument("unknown execution mode: " + value);
}

std::string execution_mode_string(ExecutionMode mode) {
    switch (mode) {
    case ExecutionMode::DataParallel:
        return "data_parallel";
    case ExecutionMode::PipelineParallel:
        return "pipeline_parallel";
    case ExecutionMode::TensorParallel:
        return "tensor_parallel";
    }

    return "data_parallel";
}

} // namespace jetsonfabric::runtime
