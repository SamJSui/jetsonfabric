#pragma once

#include "adapters/llama_cpp_adapter.hpp"
#include "pipeline_parallel/layer_executor.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

// LlamaCppFullModelExecutor is the compatibility executor used until llama.cpp
// can produce and consume partial-layer activation tensors.
class LlamaCppFullModelExecutor final : public LayerExecutor {
public:
    explicit LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter);

    inference::ExecutionResult execute(const inference::StageInput& input) const override;

private:
    adapters::LlamaCppAdapter& adapter_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
