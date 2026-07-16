#include "pipeline_parallel/llama_cpp_stage_executor.hpp"

#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {

LlamaCppStageExecutor::LlamaCppStageExecutor(adapters::LlamaCppStageConfig config)
    : adapter_(std::move(config)) {}

inference::ExecutionResult LlamaCppStageExecutor::execute(const inference::StageInput& input) const {
    return adapter_.execute(input);
}

void LlamaCppStageExecutor::close_session(const std::string& session_id) const {
    adapter_.close_session(session_id);
}

std::size_t LlamaCppStageExecutor::session_count() const {
    return adapter_.session_count();
}

} // namespace jetsonfabric::runtime::pipeline_parallel
