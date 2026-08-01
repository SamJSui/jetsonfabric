#include "pipeline_parallel/llama_cpp_stage_executor.hpp"

#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {

LlamaCppStageExecutor::LlamaCppStageExecutor(adapters::LlamaCppStageConfig config)
    : adapter_(std::move(config)) {}

inference::ExecutionResult LlamaCppStageExecutor::execute(const inference::StageInput& input) const {
    return adapter_.execute(input);
}

std::vector<inference::ExecutionResult> LlamaCppStageExecutor::execute_batch(
    const std::vector<inference::StageInput>& inputs
) const {
    return adapter_.execute_batch(inputs);
}

void LlamaCppStageExecutor::close_session(const std::string& session_id) const {
    adapter_.close_session(session_id);
}

void LlamaCppStageExecutor::rollback_session(const std::string& session_id, int token_count) const {
    adapter_.rollback_session(session_id, token_count);
}

std::size_t LlamaCppStageExecutor::session_count() const {
    return adapter_.session_count();
}

} // namespace jetsonfabric::runtime::pipeline_parallel
