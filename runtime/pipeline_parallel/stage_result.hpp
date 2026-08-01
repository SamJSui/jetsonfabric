#pragma once

#include "protocol/stage.hpp"

#include <cstdint>
#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

struct StageRunResult {
    bool ok = false;
    std::string status = "500 Internal Server Error";
    std::string error_code;
    std::string error_message;
    protocol::StageResponse response;
    std::int64_t remote_call_us = 0;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
