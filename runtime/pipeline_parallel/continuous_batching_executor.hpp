#pragma once

#include "inference/executor.hpp"

#include <chrono>
#include <cstddef>
#include <memory>

namespace jetsonfabric::runtime::pipeline_parallel {

struct ContinuousBatchingConfig {
    std::size_t max_batch_size = 1;
    std::chrono::microseconds max_wait = std::chrono::microseconds(500);
};

// Coalesces concurrently arriving decode steps. Prefill remains immediate so a
// long prompt cannot delay active decoders. Engine adapters decide how a batch
// maps to their native multi-sequence execution API.
class ContinuousBatchingExecutor final : public inference::Executor {
public:
    ContinuousBatchingExecutor(
        std::unique_ptr<inference::Executor> delegate,
        ContinuousBatchingConfig config
    );
    ~ContinuousBatchingExecutor() override;

    ContinuousBatchingExecutor(const ContinuousBatchingExecutor&) = delete;
    ContinuousBatchingExecutor& operator=(const ContinuousBatchingExecutor&) = delete;

    inference::ExecutionResult execute(const inference::StageInput& input) const override;
    std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>& inputs
    ) const override;
    void close_session(const std::string& session_id) const override;
    void rollback_session(const std::string& session_id, int token_count) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
