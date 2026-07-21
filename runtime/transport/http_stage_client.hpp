#pragma once

#include "pipeline_parallel/generation_runner.hpp"

#include <string>
#include <utility>

namespace jetsonfabric::runtime::transport {

class HTTPStageClient {
public:
    explicit HTTPStageClient(std::string cluster_token)
        : cluster_token_(std::move(cluster_token)) {}

    pipeline_parallel::StageRunResult invoke(
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& request,
        pipeline_parallel::StageOperation operation
    ) const;

private:
    std::string cluster_token_;
};

} // namespace jetsonfabric::runtime::transport
