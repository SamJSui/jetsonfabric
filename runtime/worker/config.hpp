#pragma once

#include "pipeline_parallel/stage_assignment.hpp"
#include "protocol/execution_mode.hpp"
#include "protocol/kv_cache_type.hpp"

#include <string>

namespace jetsonfabric::runtime {

struct Config {
    std::string host = "127.0.0.1";
    int port = 9090;

    std::string node_name = "runtime";
    std::string model = "runtime-model";
    std::string cluster_token;

    std::string engine = "llama.cpp";
    std::string stage_transport = "http_binary_v1";
    std::string activation_encoding = "f32";
    std::string compute_backend = "cuda";
    std::string model_path;
    int ctx_size = 4096;
    int ubatch_size = 512;
    KVCacheType kv_cache_type = KVCacheType::F16;
    int n_gpu_layers = 999;
    int threads = 0;
    int http_workers = 2;
    int parallel_sessions = 2;
    int decode_batch_size = 1;
    std::string speculative_draft = "none";
    int speculative_max_tokens = 4;

    ExecutionMode mode = ExecutionMode::DataParallel;
    pipeline_parallel::StageAssignment stage_assignment;

    bool start_idle = false;
};

Config parse_args(int argc, char** argv);
void validate_runtime_config(const Config& config);
void validate_deployment_config(const Config& config);
void print_help();

} // namespace jetsonfabric::runtime
