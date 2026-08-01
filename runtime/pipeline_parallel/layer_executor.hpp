#pragma once

#include "inference/stage.hpp"

#include <string>
#include <stdexcept>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {

class LayerExecutor {
public:
    virtual ~LayerExecutor() = default;

    virtual inference::ExecutionResult execute(const inference::StageInput& input) const = 0;

    virtual std::vector<inference::ExecutionResult> execute_batch(
        const std::vector<inference::StageInput>& inputs
    ) const {
        std::vector<inference::ExecutionResult> results;
        results.reserve(inputs.size());
        for (const inference::StageInput& input : inputs) {
            results.push_back(execute(input));
        }
        return results;
    }

    virtual void close_session(const std::string& session_id) const {
        (void) session_id;
    }

    virtual void rollback_session(const std::string& session_id, int token_count) const {
        (void) session_id;
        (void) token_count;
        throw std::runtime_error("layer executor does not support KV-cache rollback");
    }
};

} // namespace jetsonfabric::runtime::pipeline_parallel
