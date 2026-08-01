#pragma once

#include "adapters/llama_cpp_stage_adapter.hpp"
#include "inference/executor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {

class LlamaCppStageExecutor final : public inference::Executor {
public:
    explicit LlamaCppStageExecutor(adapters::LlamaCppStageConfig config);

    inference::ExecutionResult execute(const inference::StageInput& input) const override;
    std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>& inputs
    ) const override;
    void close_session(const std::string& session_id) const override;
    void rollback_session(const std::string& session_id, int token_count) const override;
    std::size_t session_count() const;

private:
    adapters::LlamaCppStageAdapter adapter_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
