#pragma once

#include "inference/stage.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::inference {

// Executes the engine-owned portion of an inference request. Pipeline stages
// and full-model strategies share this contract; topology-specific routing
// stays outside the engine factory.
class Executor {
public:
    virtual ~Executor() = default;

    virtual ExecutionResult execute(const StageInput& input) const = 0;

    virtual std::vector<ExecutionResult> execute_batch(
        const std::vector<StageInput>& inputs
    ) const {
        std::vector<ExecutionResult> results;
        results.reserve(inputs.size());
        for (const StageInput& input : inputs) {
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
        throw std::runtime_error("inference executor does not support KV-cache rollback");
    }
};

} // namespace jetsonfabric::runtime::inference
