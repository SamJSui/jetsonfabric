#pragma once

#include "transport/stage_transport.hpp"

#include <memory>
#include <string>

namespace jetsonfabric::runtime::transport {

enum class HTTPStageTarget {
    NodeAPI,
    Runtime,
};

class HTTPStageTransport final : public StageTransport {
public:
    explicit HTTPStageTransport(
        std::string cluster_token,
        HTTPStageTarget target = HTTPStageTarget::NodeAPI
    );
    ~HTTPStageTransport() override;

    void shutdown() const override;
    pipeline_parallel::StageRunResult invoke(
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& request,
        pipeline_parallel::StageOperation operation
    ) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::transport
