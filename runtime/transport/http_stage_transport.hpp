#pragma once

#include "transport/stage_transport.hpp"

#include <string>
#include <utility>

namespace jetsonfabric::runtime::transport {

class HTTPStageTransport final : public StageTransport {
public:
    explicit HTTPStageTransport(std::string cluster_token)
        : cluster_token_(std::move(cluster_token)) {}

    pipeline_parallel::StageRunResult invoke(
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& request,
        pipeline_parallel::StageOperation operation
    ) const override;

private:
    std::string cluster_token_;
};

} // namespace jetsonfabric::runtime::transport
