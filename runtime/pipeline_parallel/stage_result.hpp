#pragma once

#include "protocol/stage.hpp"

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

struct StageRunResult {
    bool ok = false;
    std::string status = "500 Internal Server Error";
    std::string error_code;
    std::string error_message;
    protocol::StageResponse response;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
