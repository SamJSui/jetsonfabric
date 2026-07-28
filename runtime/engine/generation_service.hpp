#pragma once

#include "deployment/model_manager.hpp"
#include "pipeline_parallel/generation_runner.hpp"
#include "protocol/execution_mode.hpp"
#include "transport/stage_transport.hpp"

#include <optional>
#include <string>

namespace jetsonfabric::runtime {

// Owns one runtime's generation use case. It validates the selected deployment,
// invokes stage zero locally, and delegates peer stages to StageTransport.
class GenerationService {
public:
    GenerationService(
        std::string node_name,
        ExecutionMode execution_mode,
        const deployment::ModelManager& model_manager,
        const transport::StageTransport& stage_transport
    );

    pipeline_parallel::GenerationResult generate(
        const protocol::GenerationRequest& request,
        const pipeline_parallel::TokenSink& sink
    ) const;

private:
    const deployment::DeploymentIdentity* select_deployment(
        const protocol::GenerationRequest& request
    ) const;
    std::optional<pipeline_parallel::GenerationResult> validate_request(
        const protocol::GenerationRequest& request,
        const deployment::DeploymentIdentity* deployment
    ) const;
    pipeline_parallel::StageRunResult invoke_stage(
        const protocol::GenerationStage& stage,
        const protocol::StageRequest& request,
        pipeline_parallel::StageOperation operation
    ) const;

    std::string node_name_;
    ExecutionMode execution_mode_;
    const deployment::ModelManager& model_manager_;
    const transport::StageTransport& stage_transport_;
};

} // namespace jetsonfabric::runtime
