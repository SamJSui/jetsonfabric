#pragma once

#include "adapters/llama_cpp_adapter.hpp"
#include "pipeline_parallel/layer_executor.hpp"

namespace jetsonfabric::runtime::pipeline_parallel {

class LlamaCppFullModelExecutor final : public LayerExecutor {
public:
    explicit LlamaCppFullModelExecutor(adapters::LlamaCppAdapter& adapter);

    StageRunResult run_layers(const protocol::ActivationRequest& request) const override;

private:
    adapters::LlamaCppAdapter& adapter_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel