#pragma once

#include "pipeline_parallel/stage_result.hpp"
#include "protocol/activation.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

class LayerExecutor {
public:
    virtual ~LayerExecutor() = default;

    virtual StageRunResult run_layers(const protocol::ActivationRequest& request) const = 0;
};

class UnavailableLayerExecutor final : public LayerExecutor {
public:
    StageRunResult run_layers(const protocol::ActivationRequest& request) const override;
};

} // namespace jetsonfabric::runtime::pipeline_parallel