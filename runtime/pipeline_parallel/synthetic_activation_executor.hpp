#pragma once

#include "pipeline_parallel/layer_executor.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

// SyntheticActivationExecutor is a deterministic CI/test engine. It exercises
// the real stagewire and logical-node data path without claiming to execute
// transformer layers.
class SyntheticActivationExecutor final : public LayerExecutor {
public:
    StageRunResult run_layers(const protocol::StageRequest& request) const override;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
