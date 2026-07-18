#pragma once

#include "inference/stage.hpp"

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

class LayerExecutor {
public:
    virtual ~LayerExecutor() = default;

    virtual inference::ExecutionResult execute(const inference::StageInput& input) const = 0;

    virtual void close_session(const std::string& session_id) const {
        (void) session_id;
    }
};

} // namespace jetsonfabric::runtime::pipeline_parallel
