#pragma once

#include "adapters/llama_cpp_adapter.hpp"
#include "inference/executor.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

// LlamaCppFullModelExecutor is the compatibility executor used until llama.cpp
// can produce and consume partial-layer activation tensors.
class LlamaCppFullModelExecutor final : public inference::Executor {
public:
    explicit LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter);

    inference::ExecutionResult execute(const inference::StageInput& input) const override;

private:
    adapters::LlamaCppAdapter& adapter_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
