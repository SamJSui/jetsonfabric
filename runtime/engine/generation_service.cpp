#include "engine/generation_service.hpp"

#include <utility>

namespace jetsonfabric::runtime {
namespace {

pipeline_parallel::GenerationResult generation_error(
    std::string code,
    std::string message
) {
    pipeline_parallel::GenerationResult result;
    result.error_code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

} // namespace

GenerationService::GenerationService(
    std::string node_name,
    ExecutionMode execution_mode,
    const deployment::ModelManager& model_manager,
    const transport::StageTransport& stage_transport
)
    : node_name_(std::move(node_name)),
      execution_mode_(execution_mode),
      model_manager_(model_manager),
      stage_transport_(stage_transport) {}

pipeline_parallel::GenerationResult GenerationService::generate(
    const protocol::GenerationRequest& request,
    const pipeline_parallel::TokenSink& sink
) const {
    const std::optional<deployment::DeploymentIdentity> selected =
        select_deployment(request);
    if (const auto validation_error = validate_request(request, selected);
        validation_error.has_value()) {
        return *validation_error;
    }

    pipeline_parallel::GenerationRunner runner(
        [this](
            const protocol::GenerationStage& stage,
            const protocol::StageRequest& stage_request,
            pipeline_parallel::StageOperation operation
        ) {
            return invoke_stage(stage, stage_request, operation);
        }
    );
    return runner.run(request, sink);
}

std::optional<deployment::DeploymentIdentity> GenerationService::select_deployment(
    const protocol::GenerationRequest& request
) const {
    return request.deployment.has_value()
        ? model_manager_.executable_deployment_identity(*request.deployment)
        : model_manager_.active_deployment_identity();
}

std::optional<pipeline_parallel::GenerationResult> GenerationService::validate_request(
    const protocol::GenerationRequest& request,
    const std::optional<deployment::DeploymentIdentity>& selected
) const {
    if (execution_mode_ != ExecutionMode::PipelineParallel) {
        return generation_error(
            "invalid_execution_mode",
            "runtime-owned generation requires pipeline_parallel mode"
        );
    }
    if (!selected.has_value()) {
        const bool identity_mismatch =
            request.deployment.has_value() && model_manager_.has_active_deployment();
        return generation_error(
            identity_mismatch ? "deployment_mismatch" : "no_active_deployment",
            identity_mismatch
                ? "generation deployment identity does not match an executable runtime epoch"
                : "runtime has no executable deployment for the requested epoch"
        );
    }
    if (selected->model_id != request.model_id) {
        return generation_error(
            "deployment_mismatch",
            "generation model does not match the selected deployment"
        );
    }
    if (selected->epoch > 0 && !request.deployment.has_value()) {
        return generation_error(
            "deployment_identity_required",
            "managed runtime generation requires deployment identity"
        );
    }
    if (request.deployment.has_value() && *request.deployment != *selected) {
        return generation_error(
            "deployment_mismatch",
            "generation deployment identity does not match an executable runtime epoch"
        );
    }
    if (request.stages.empty() ||
        request.stages.front().stage_index != 0 ||
        request.stages.front().node_name != node_name_) {
        return generation_error(
            "invalid_pipeline_leader",
            "generation must be sent to the runtime assigned stage zero"
        );
    }
    return std::nullopt;
}

pipeline_parallel::StageRunResult GenerationService::invoke_stage(
    const protocol::GenerationStage& stage,
    const protocol::StageRequest& request,
    pipeline_parallel::StageOperation operation
) const {
    if (stage.stage_index == 0) {
        return operation == pipeline_parallel::StageOperation::CloseSession
            ? model_manager_.close_session(request)
            : model_manager_.run_stage(request);
    }
    return stage_transport_.invoke(stage, request, operation);
}

} // namespace jetsonfabric::runtime
