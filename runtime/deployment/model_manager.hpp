#pragma once

#include "deployment/deployment.hpp"
#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::runtime::deployment {

// ModelManager owns the runtime's active deployment execution components. The
// configured startup path still installs one deployment immediately, while an
// empty manager provides the idle state needed by later lifecycle operations.
class ModelManager {
public:
    ModelManager() = default;

    explicit ModelManager(const Config& config)
        : ModelManager(
              config.node_name,
              DeploymentIdentity{
                  .deployment_id = config.model,
                  .model_id = config.model,
              },
              config.stage_assignment,
              build_inference_engine_parts(config)
          ) {}

    ModelManager(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        InferenceEngineParts engine_parts
    )
        : active_(
              std::in_place,
              std::move(node_name),
              std::move(identity),
              assignment,
              std::move(engine_parts)
          ) {}

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;

    bool has_active_deployment() const noexcept {
        return active_.has_value();
    }

    const DeploymentIdentity* active_deployment_identity() const noexcept {
        return active_ ? &active_->identity : nullptr;
    }

    std::optional<DeploymentState> active_deployment_state() const noexcept {
        return active_ ? std::optional<DeploymentState>{active_->state} : std::nullopt;
    }

    const std::string& active_deployment_id() const noexcept {
        static const std::string empty_deployment_id;
        return active_ ? active_->identity.deployment_id : empty_deployment_id;
    }

    const std::string& active_model_id() const noexcept {
        static const std::string empty_model_id;
        return active_ ? active_->identity.model_id : empty_model_id;
    }

    pipeline_parallel::StageRunResult run_stage(const protocol::StageRequest& request) const {
        if (!active_) {
            return no_active_deployment();
        }
        return active_->stage_worker.run(request);
    }

    pipeline_parallel::StageRunResult close_session(const protocol::StageRequest& request) const {
        if (!active_) {
            return no_active_deployment();
        }
        return active_->stage_worker.close_session(request);
    }

private:
    static pipeline_parallel::StageRunResult no_active_deployment() {
        pipeline_parallel::StageRunResult result;
        result.ok = false;
        result.status = "503 Service Unavailable";
        result.error_code = "no_active_deployment";
        result.error_message = "runtime has no active deployment";
        return result;
    }

    static DeploymentIdentity require_identity(DeploymentIdentity identity) {
        if (identity.deployment_id.empty()) {
            throw std::invalid_argument("deployment_id is required");
        }
        if (identity.model_id.empty()) {
            throw std::invalid_argument("model_id is required");
        }
        return identity;
    }

    static const pipeline_parallel::LayerExecutor& require_layer_executor(
        const InferenceEngineParts& engine_parts
    ) {
        if (!engine_parts.layer_executor) {
            throw std::invalid_argument("model manager requires a layer executor");
        }
        return *engine_parts.layer_executor;
    }

    struct Deployment {
        Deployment(
            std::string node_name,
            DeploymentIdentity deployment_identity,
            pipeline_parallel::StageAssignment assignment,
            InferenceEngineParts loaded_engine_parts
        )
            : identity(ModelManager::require_identity(std::move(deployment_identity))),
              state(DeploymentState::Active),
              engine_parts(std::move(loaded_engine_parts)),
              stage_worker(
                  std::move(node_name),
                  identity.model_id,
                  assignment,
                  ModelManager::require_layer_executor(engine_parts)
              ) {}

        Deployment(const Deployment&) = delete;
        Deployment& operator=(const Deployment&) = delete;
        Deployment(Deployment&&) = delete;
        Deployment& operator=(Deployment&&) = delete;

        DeploymentIdentity identity;
        DeploymentState state;
        InferenceEngineParts engine_parts;
        pipeline_parallel::StageWorker stage_worker;
    };

    std::optional<Deployment> active_;
};

} // namespace jetsonfabric::runtime::deployment
