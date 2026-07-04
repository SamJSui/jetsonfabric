#pragma once

#include "pipeline_parallel/layer_executor.hpp"
#include "pipeline_parallel/stage_assignment.hpp"
#include "pipeline_parallel/stage_result.hpp"
#include "protocol/activation.hpp"

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

class StageWorker {
public:
    StageWorker(
        std::string node_name,
        StageAssignment assignment,
        const LayerExecutor& layer_executor
    );

    StageRunResult run(const protocol::ActivationRequest& request) const;

private:
    std::string node_name_;
    StageAssignment assignment_;
    const LayerExecutor& layer_executor_;

    std::string validate_request(const protocol::ActivationRequest& request) const;
};

} // namespace jetsonfabric::runtime::pipeline_parallel