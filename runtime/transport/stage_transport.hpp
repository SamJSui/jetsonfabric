#pragma once

#include "pipeline_parallel/generation_runner.hpp"

namespace jetsonfabric::runtime::transport {

// Carries one stage operation to a runtime on another node.
// Planning and local execution remain outside this boundary.
class StageTransport {
public:
    virtual ~StageTransport() = default;

    virtual pipeline_parallel::StageRunResult invoke(
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& request,
        pipeline_parallel::StageOperation operation
    ) const = 0;
};

} // namespace jetsonfabric::runtime::transport
