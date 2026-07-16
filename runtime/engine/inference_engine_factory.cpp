#include "engine/inference_engine_factory.hpp"

#include "adapters/llama_cpp_adapter.hpp"
#include "pipeline_parallel/llama_cpp_full_model_executor.hpp"
#include "pipeline_parallel/synthetic_activation_executor.hpp"

#include <memory>
#include <stdexcept>

namespace jetsonfabric::runtime {
namespace {

class LlamaCppExecutorHolder final : public pipeline_parallel::LayerExecutor {
public:
    explicit LlamaCppExecutorHolder(const Config& config)
        : adapter_(adapters::LlamaCppConfig{
              .model_path = config.model_path,
              .ctx_size = config.ctx_size,
              .n_gpu_layers = config.compute_backend == "cuda" ? config.n_gpu_layers : 0,
              .threads = config.threads,
          }),
          executor_(adapter_) {}

    pipeline_parallel::StageRunResult run_layers(const protocol::StageRequest& request) const override {
        return executor_.run_layers(request);
    }

private:
    adapters::LlamaCppAdapter adapter_;
    pipeline_parallel::LlamaCppFullModelExecutor executor_;
};

} // namespace

InferenceEngineParts build_inference_engine_parts(const Config& config) {
    if (config.engine == "llama.cpp") {
        return InferenceEngineParts{.layer_executor = std::make_unique<LlamaCppExecutorHolder>(config)};
    }
    if (config.engine == "synthetic") {
        return InferenceEngineParts{.layer_executor = std::make_unique<pipeline_parallel::SyntheticActivationExecutor>()};
    }
    throw std::invalid_argument("unsupported inference engine: " + config.engine);
}

} // namespace jetsonfabric::runtime
