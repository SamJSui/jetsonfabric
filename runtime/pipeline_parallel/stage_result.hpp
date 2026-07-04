#pragma once

#include "protocol/activation.hpp"

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

struct StageRunResult {
    bool ok = false;
    std::string status = "500 Internal Server Error";
    std::string error_code;
    std::string error_message;
    protocol::ActivationResponse response;
};

} // namespace jetsonfabric::runtime::pipeline_parallel