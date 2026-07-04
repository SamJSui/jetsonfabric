#pragma once

#include "protocol/execution_mode.hpp"

#include <string>

namespace jetsonfabric::runtime {

struct EngineResponse {
    std::string status;
    std::string body;
};

class Engine {
public:
    virtual ~Engine() = default;

    virtual std::string runtime_name() const = 0;
    virtual ExecutionMode mode() const = 0;
    virtual std::string model() const = 0;

    virtual EngineResponse chat_completion(const std::string& request_body) const = 0;
    virtual EngineResponse run_stage(const std::string& request_body) const = 0;
};

} // namespace jetsonfabric::runtime