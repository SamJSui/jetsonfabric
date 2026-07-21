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
    std::string cluster_token;

    std::string engine = "llama.cpp";
    std::string compute_backend = "cuda";
    std::string model_path;
    int ctx_size = 4096;
    int n_gpu_layers = 999;
    int threads = 0;

    ExecutionMode mode = ExecutionMode::DataParallel;
    pipeline_parallel::StageAssignment stage_assignment;

    bool start_idle = false;
};

Config parse_args(int argc, char** argv);
void validate_runtime_config(const Config& config);
void validate_deployment_config(const Config& config);
void print_help();

} // namespace jetsonfabric::runtime
