#pragma once

#include "pipeline_parallel/stage_result.hpp"
#include "protocol/stage.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

class LayerExecutor {
public:
    virtual ~LayerExecutor() = default;

    virtual StageRunResult run_layers(const protocol::StageRequest& request) const = 0;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
