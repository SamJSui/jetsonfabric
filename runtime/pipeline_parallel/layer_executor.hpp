#pragma once

#include "inference/stage.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

class LayerExecutor {
public:
    virtual ~LayerExecutor() = default;

    virtual inference::ExecutionResult execute(const inference::StageInput& input) const = 0;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
