#pragma once

#include "activation/activation_codec.hpp"
#include "inference/executor.hpp"
#include "pipeline_parallel/stage_assignment.hpp"
#include "pipeline_parallel/stage_result.hpp"
#include "protocol/stage.hpp"

#include <memory>
#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

class StageWorker {
public:
    StageWorker(
        std::string node_name,
        std::string model_id,
        StageAssignment assignment,
        const inference::Executor& executor,
        std::shared_ptr<const activation::ActivationCodec> activation_codec
    );

    StageRunResult run(const protocol::StageRequest& request) const;
    StageRunResult close_session(const protocol::StageRequest& request) const;
    StageRunResult rollback_session(const protocol::StageRequest& request) const;

private:
    std::string node_name_;
    std::string model_id_;
    StageAssignment assignment_;
    const inference::Executor& executor_;
    std::shared_ptr<const activation::ActivationCodec> activation_codec_;

    std::string validate_request(const protocol::StageRequest& request) const;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
