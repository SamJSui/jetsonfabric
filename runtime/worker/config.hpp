#pragma once

#include "pipeline_parallel/stage_assignment.hpp"
#include "protocol/execution_mode.hpp"

#include <string>

namespace jetsonfabric::runtime {

struct Config {
    std::string host = "127.0.0.1";
    int port = 9090;

    std::string node_name = "runtime";
    std::string model = "runtime-model";

    ExecutionMode mode = ExecutionMode::DataParallel;
    pipeline_parallel::StageAssignment stage_assignment;
};

Config parse_args(int argc, char** argv);
void print_help();

} // namespace jetsonfabric::runtime