#pragma once

#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::runtime::deployment {

// ModelManager owns the runtime's active model execution components. For now a
// runtime still starts with exactly one active model; later deployment changes
// can replace that model without changing RuntimeService's request path.
class ModelManager {
public:
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
        : active_(std::make_unique<LoadedModel>(
              std::move(node_name),
              std::move(model_id),
              assignment,
              std::move(engine_parts)
          )) {}

    const std::string& active_model_id() const noexcept {
        return active_->model_id;
    }

    pipeline_parallel::StageRunResult run_stage(const protocol::StageRequest& request) const {
        return active_->stage_worker.run(request);
    }

    pipeline_parallel::StageRunResult close_session(const protocol::StageRequest& request) const {
        return active_->stage_worker.close_session(request);
    }

private:
    static const pipeline_parallel::LayerExecutor& require_layer_executor(
        const InferenceEngineParts& engine_parts
    ) {
        if (!engine_parts.layer_executor) {
            throw std::invalid_argument("model manager requires a layer executor");
        }
        return *engine_parts.layer_executor;
    }

    struct LoadedModel {
        LoadedModel(
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

        std::string model_id;
        InferenceEngineParts engine_parts;
        pipeline_parallel::StageWorker stage_worker;
    };

    std::unique_ptr<LoadedModel> active_;
};

} // namespace jetsonfabric::runtime::deployment
