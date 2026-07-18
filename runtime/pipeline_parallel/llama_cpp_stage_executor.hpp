#pragma once

#include "adapters/llama_cpp_stage_adapter.hpp"
#include "pipeline_parallel/layer_executor.hpp"

#include <cstddef>
#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

class LlamaCppStageExecutor final : public LayerExecutor {
public:
    explicit LlamaCppStageExecutor(adapters::LlamaCppStageConfig config);

    inference::ExecutionResult execute(const inference::StageInput& input) const override;
    void close_session(const std::string& session_id) const override;
    std::size_t session_count() const;

private:
    adapters::LlamaCppStageAdapter adapter_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
