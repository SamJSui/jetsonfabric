#pragma once

#include "pipeline_parallel/layer_executor.hpp"
#include "pipeline_parallel/stage_assignment.hpp"
#include "pipeline_parallel/stage_result.hpp"
#include "protocol/stage.hpp"

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

class StageWorker {
public:
    StageWorker(
        std::string node_name,
        std::string model_id,
        StageAssignment assignment,
        const LayerExecutor& layer_executor
    );

    StageRunResult run(const protocol::StageRequest& request) const;
    StageRunResult close_session(const protocol::StageRequest& request) const;

private:
    std::string node_name_;
    std::string model_id_;
    StageAssignment assignment_;
    const LayerExecutor& layer_executor_;

    std::string validate_request(const protocol::StageRequest& request) const;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
