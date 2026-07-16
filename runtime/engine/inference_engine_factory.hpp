#pragma once

#include "pipeline_parallel/layer_executor.hpp"
#include "worker/config.hpp"

#include <memory>

namespace jetsonfabric::runtime {

// InferenceEngineParts contains the adapter-backed execution components hosted
// by RuntimeService. The runtime process itself is not an inference engine.
struct InferenceEngineParts {
    std::unique_ptr<pipeline_parallel::LayerExecutor> layer_executor;
};

InferenceEngineParts build_inference_engine_parts(const Config& config);

} // namespace jetsonfabric::runtime
