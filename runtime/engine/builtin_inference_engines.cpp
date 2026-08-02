#include "engine/inference_engine_factory.hpp"

#include "adapters/llama_cpp_adapter.hpp"
#include "adapters/llama_cpp_model.hpp"
#include "pipeline_parallel/continuous_batching_executor.hpp"
#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"
#include "pipeline_parallel/llama_cpp_stage_executor.hpp"
#include "pipeline_parallel/synthetic_activation_executor.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime {
namespace {

bool uses_stage_execution(ExecutionMode mode) {
    return mode == ExecutionMode::PipelineParallel ||
        mode == ExecutionMode::TensorParallel;
}

std::shared_ptr<adapters::LlamaCppModel> load_llama_model(const Config& config) {
    const bool pipeline = config.mode == ExecutionMode::PipelineParallel;
    return std::make_shared<adapters::LlamaCppModel>(adapters::LlamaCppModelConfig{
        .model_path = config.model_path,
        .n_gpu_layers = config.compute_backend == "cuda" ? config.n_gpu_layers : 0,
        .layer_start = pipeline ? config.stage_assignment.layer_start : 0,
        .layer_end = pipeline ? config.stage_assignment.layer_end : 0,
        .tensor_parallel = config.mode == ExecutionMode::TensorParallel
            ? std::optional<tensor_parallel::DeviceMesh>(config.tensor_mesh)
            : std::nullopt,
    });
}

deployment::ModelResidency describe_residency(const adapters::LlamaCppModel& model) {
    return deployment::ModelResidency{
        .layer_start = model.loaded_layer_start(),
        .layer_end = model.loaded_layer_end(),
        .layer_count = model.n_layer(),
        .resident_weight_bytes = model.resident_weight_bytes(),
        .total_weight_bytes = model.total_weight_bytes(),
        .resident_tensor_count = model.resident_tensor_count(),
    };
}

class LlamaCppFullModelAdapter final : public inference::Executor {
public:
    LlamaCppFullModelAdapter(
        std::shared_ptr<adapters::LlamaCppModel> model,
        const Config& config
    )
        : model_(std::move(model)),
          adapter_(
              model_,
              config.ctx_size,
              config.threads,
              config.kv_cache_type,
              config.ubatch_size
          ),
          executor_(adapter_) {}

    inference::ExecutionResult execute(const inference::StageInput& input) const override {
        return executor_.execute(input);
    }

private:
    std::shared_ptr<adapters::LlamaCppModel> model_;
    adapters::LlamaCppAdapter adapter_;
    pipeline_parallel::LlamaCppFullModelExecutor executor_;
};

InferenceEngineParts create_llama_cpp_engine(const Config& config) {
    if (config.model_path.empty()) {
        throw std::invalid_argument("llama.cpp requires a model path");
    }
    if (config.compute_backend != "cpu" && config.compute_backend != "cuda") {
        throw std::invalid_argument("llama.cpp compute backend must be cpu or cuda");
    }
    std::shared_ptr<adapters::LlamaCppModel> model = load_llama_model(config);
    const deployment::ModelResidency residency = describe_residency(*model);
    if (uses_stage_execution(config.mode)) {
        const bool tensor_parallel = config.mode == ExecutionMode::TensorParallel;
        const inference::StagePosition position = tensor_parallel
            ? inference::StagePosition{.index = 0, .count = 1}
            : inference::StagePosition{
                  .index = config.stage_assignment.stage_index,
                  .count = config.stage_assignment.stage_count,
              };
        const inference::LayerRange layers = tensor_parallel
            ? inference::LayerRange{.start = 0, .end = model->n_layer()}
            : inference::LayerRange{
                  .start = config.stage_assignment.layer_start,
                  .end = config.stage_assignment.layer_end,
              };
        std::unique_ptr<inference::Executor> executor =
            std::make_unique<pipeline_parallel::LlamaCppStageExecutor>(
                adapters::LlamaCppStageConfig{
                    .model = std::move(model),
                    .ctx_size = config.ctx_size,
                    .ubatch_size = config.ubatch_size,
                    .decode_batch_size = config.decode_batch_size,
                    .max_parallel_sessions = config.parallel_sessions,
                    .speculative_max_tokens = config.speculative_draft == "none"
                        ? 0
                        : config.speculative_max_tokens,
                    .kv_cache_type = config.kv_cache_type,
                    .threads = config.threads,
                    .position = position,
                    .layers = layers,
                }
            );
        if (config.decode_batch_size > 1) {
            executor = std::make_unique<pipeline_parallel::ContinuousBatchingExecutor>(
                std::move(executor),
                pipeline_parallel::ContinuousBatchingConfig{
                    .max_batch_size = static_cast<std::size_t>(config.decode_batch_size),
                }
            );
        }
        return InferenceEngineParts{
            .executor = std::move(executor),
            .model_residency = residency,
            .execution_assignment = pipeline_parallel::StageAssignment{
                .stage_index = position.index,
                .stage_count = position.count,
                .layer_start = layers.start,
                .layer_end = layers.end,
            },
        };
    }
    return InferenceEngineParts{
        .executor = std::make_unique<LlamaCppFullModelAdapter>(std::move(model), config),
        .model_residency = residency,
        .execution_assignment = std::nullopt,
    };
}

InferenceEngineParts create_synthetic_engine(const Config&) {
    return InferenceEngineParts{
        .executor = std::make_unique<pipeline_parallel::SyntheticActivationExecutor>(),
        .model_residency = std::nullopt,
        .execution_assignment = std::nullopt,
    };
}

} // namespace

std::shared_ptr<const InferenceEngineFactory> make_default_inference_engine_factory() {
    auto factory = std::make_shared<InferenceEngineFactory>();
    factory->register_engine("llama.cpp", create_llama_cpp_engine);
    factory->register_engine("synthetic", create_synthetic_engine);
    return factory;
}

} // namespace jetsonfabric::runtime
