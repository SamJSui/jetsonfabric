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
    return std::make_shared<adapters::LlamaCppModel>(adapters::LlamaCppModelConfig{
        .model_path = config.model_path,
        .n_gpu_layers = config.compute_backend == "cuda" ? config.n_gpu_layers : 0,
    });
}

class LlamaCppCompatibilityHolder final : public pipeline_parallel::LayerExecutor {
public:
    explicit LlamaCppCompatibilityHolder(const Config& config)
        : model_(make_llama_model(config)),
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
        if (config.mode == ExecutionMode::PipelineParallel && config.stage_assignment.stage_count > 1) {
            return InferenceEngineParts{
                .layer_executor = std::make_unique<pipeline_parallel::LlamaCppStageExecutor>(
                    adapters::LlamaCppStageConfig{
                        .model = make_llama_model(config),
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
            };
        }
        return InferenceEngineParts{
            .layer_executor = std::make_unique<LlamaCppCompatibilityHolder>(config),
        };
    }
    if (config.engine == "synthetic") {
        return InferenceEngineParts{
            .layer_executor = std::make_unique<pipeline_parallel::SyntheticActivationExecutor>(),
        };
    }
    throw std::invalid_argument("unsupported inference engine: " + config.engine);
}

} // namespace jetsonfabric::runtime
