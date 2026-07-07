#pragma once

#include "pipeline_parallel/layer_executor.hpp"
#include "worker/config.hpp"

#include <memory>

namespace jetsonfabric::runtime {

struct RuntimeEngineParts {
    std::unique_ptr<pipeline_parallel::LayerExecutor> layer_executor;
};

RuntimeEngineParts build_runtime_engine_parts(const Config& config);

} // namespace jetsonfabric::runtime