#pragma once

#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
              config.model,
              config.stage_assignment,
              build_inference_engine_parts(config)
          ) {}

    ModelManager(
        std::string node_name,
        std::string model_id,
        pipeline_parallel::StageAssignment assignment,
        InferenceEngineParts engine_parts
    )
        : active_(
              std::in_place,
              std::move(node_name),
              std::move(model_id),
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

    std::optional<std::string_view> active_model_id() const noexcept {
        if (!active_) {
            return std::nullopt;
        }
        return std::string_view(active_->model_id);
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
            std::string loaded_model_id,
            pipeline_parallel::StageAssignment assignment,
            InferenceEngineParts loaded_engine_parts
        )
            : model_id(std::move(loaded_model_id)),
              engine_parts(std::move(loaded_engine_parts)),
              stage_worker(
                  std::move(node_name),
                  model_id,
                  assignment,
                  ModelManager::require_layer_executor(engine_parts)
              ) {}

        Deployment(const Deployment&) = delete;
        Deployment& operator=(const Deployment&) = delete;
        Deployment(Deployment&&) = delete;
        Deployment& operator=(Deployment&&) = delete;

        std::string model_id;
        InferenceEngineParts engine_parts;
        pipeline_parallel::StageWorker stage_worker;
    };

    std::optional<Deployment> active_;
};

} // namespace jetsonfabric::runtime::deployment
