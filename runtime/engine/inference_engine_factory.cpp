#include "engine/inference_engine_factory.hpp"

#include "adapters/llama_cpp_adapter.hpp"
#include "adapters/llama_cpp_model.hpp"
#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"
#include "pipeline_parallel/llama_cpp_stage_executor.hpp"
#include "pipeline_parallel/synthetic_activation_executor.hpp"

#include <memory>
#include <stdexcept>

namespace jetsonfabric::runtime {
namespace {

std::shared_ptr<adapters::LlamaCppModel> make_llama_model(const Config& config) {
    const bool pipeline = config.mode == ExecutionMode::PipelineParallel;
    return std::make_shared<adapters::LlamaCppModel>(adapters::LlamaCppModelConfig{
        .model_path = config.model_path,
        .n_gpu_layers = config.compute_backend == "cuda" ? config.n_gpu_layers : 0,
        .layer_start = pipeline ? config.stage_assignment.layer_start : 0,
        .layer_end = pipeline ? config.stage_assignment.layer_end : 0,
    });
}

deployment::ModelResidency model_residency(const adapters::LlamaCppModel& model) {
    return deployment::ModelResidency{
        .layer_start = model.loaded_layer_start(),
        .layer_end = model.loaded_layer_end(),
        .layer_count = model.n_layer(),
        .resident_weight_bytes = model.resident_weight_bytes(),
        .total_weight_bytes = model.total_weight_bytes(),
        .resident_tensor_count = model.resident_tensor_count(),
    };
}

class LlamaCppCompatibilityHolder final : public pipeline_parallel::LayerExecutor {
public:
    LlamaCppCompatibilityHolder(std::shared_ptr<adapters::LlamaCppModel> model, const Config& config)
        : model_(std::move(model)),
          adapter_(model_, config.ctx_size, config.threads),
          executor_(adapter_) {}

    inference::ExecutionResult execute(const inference::StageInput& input) const override {
        return executor_.execute(input);
    }

private:
    std::shared_ptr<adapters::LlamaCppModel> model_;
    adapters::LlamaCppAdapter adapter_;
    pipeline_parallel::LlamaCppFullModelExecutor executor_;
};

} // namespace

InferenceEngineParts build_inference_engine_parts(const Config& config) {
    if (config.engine == "llama.cpp") {
        std::shared_ptr<adapters::LlamaCppModel> model = make_llama_model(config);
        const deployment::ModelResidency residency = model_residency(*model);
        // Pipeline parallelism uses one execution contract for every cluster
        // size. A single-node cluster is stage 0/1 over the full layer range;
        // multi-node clusters use the same adapter with partial ranges.
        if (config.mode == ExecutionMode::PipelineParallel && config.stage_assignment.stage_count >= 1) {
            return InferenceEngineParts{
                .layer_executor = std::make_unique<pipeline_parallel::LlamaCppStageExecutor>(
                    adapters::LlamaCppStageConfig{
                        .model = std::move(model),
                        .ctx_size = config.ctx_size,
                        .threads = config.threads,
                        .position = inference::StagePosition{
                            .index = config.stage_assignment.stage_index,
                            .count = config.stage_assignment.stage_count,
                        },
                        .layers = inference::LayerRange{
                            .start = config.stage_assignment.layer_start,
                            .end = config.stage_assignment.layer_end,
                        },
                    }
                ),
                .model_residency = residency,
            };
        }
        return InferenceEngineParts{
            .layer_executor = std::make_unique<LlamaCppCompatibilityHolder>(std::move(model), config),
            .model_residency = residency,
        };
    }
    if (config.engine == "synthetic") {
        return InferenceEngineParts{
            .layer_executor = std::make_unique<pipeline_parallel::SyntheticActivationExecutor>(),
            .model_residency = std::nullopt,
        };
    }
    throw std::invalid_argument("unsupported inference engine: " + config.engine);
}

} // namespace jetsonfabric::runtime
