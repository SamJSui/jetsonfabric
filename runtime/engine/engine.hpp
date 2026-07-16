#pragma once

#include "protocol/execution_mode.hpp"

#include <string>

namespace jetsonfabric::runtime {

struct RuntimeResponse {
    std::string status;
    std::string content_type;
    std::string body;
};

class RuntimeAPI {
public:
    virtual ~RuntimeAPI() = default;

    virtual std::string runtime_name() const = 0;
    virtual std::string engine_name() const = 0;
    virtual ExecutionMode execution_mode() const = 0;
    virtual std::string model() const = 0;

    virtual RuntimeResponse chat_completion(const std::string& request_body) const = 0;
    virtual RuntimeResponse run_stage(const std::string& request_body) const = 0;
};

} // namespace jetsonfabric::runtime
