#include "worker/config.hpp"

#include "protocol/stage.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace jetsonfabric::runtime {
namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << "\n";
    std::exit(2);
}

std::string require_value(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) fail("missing value for " + flag);
    return argv[++index];
}

int parse_int(const std::string& value, const std::string& flag) {
    if (value.empty()) fail(flag + " must not be empty");
    try {
        std::size_t consumed = 0;
        long parsed = std::stol(value, &consumed, 10);
        if (consumed != value.size()) fail(flag + " must be an integer");
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            fail(flag + " is outside int range");
        }
        return static_cast<int>(parsed);
    } catch (const std::invalid_argument&) {
        fail(flag + " must be an integer");
    } catch (const std::out_of_range&) {
        fail(flag + " is outside int range");
    }
}

bool is_loopback_host(std::string host) {
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    return host == "localhost" || host == "127.0.0.1" || host == "::1";
}

void parse_listen(Config& cfg, const std::string& value) {
    const auto colon = value.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) fail("--listen must be host:port");
    cfg.host = value.substr(0, colon);
    cfg.port = parse_int(value.substr(colon + 1), "--listen port");
    if (cfg.port <= 0 || cfg.port > 65535) fail("--listen port must be between 1 and 65535");
}

void validate_engine_config(const Config& cfg) {
    if (cfg.engine.empty()) {
        throw std::invalid_argument("--engine must not be empty");
    }
    if (cfg.stage_transport.empty()) {
        throw std::invalid_argument("--stage-transport must not be empty");
    }
    if (cfg.stage_transport == protocol::kDirectStageTransport && cfg.cluster_token.empty()) {
        throw std::invalid_argument(
            "http_direct_v1 requires JETSONFABRIC_CLUSTER_TOKEN"
        );
    }
    if (cfg.activation_encoding.empty()) {
        throw std::invalid_argument("--activation-encoding must not be empty");
    }
    if (cfg.compute_backend.empty()) {
        throw std::invalid_argument("--compute-backend must not be empty");
    }
    if (cfg.ctx_size <= 0) {
        throw std::invalid_argument("--ctx-size must be greater than zero");
    }
    if (cfg.ubatch_size <= 0) {
        throw std::invalid_argument("--ubatch-size must be greater than zero");
    }
    if (cfg.n_gpu_layers < 0) {
        throw std::invalid_argument("--n-gpu-layers must be zero or greater");
    }
    if (cfg.threads < 0) {
        throw std::invalid_argument("--threads must be zero or greater");
    }
    if (cfg.http_workers <= 0 || cfg.http_workers > 64) {
        throw std::invalid_argument("--http-workers must be between 1 and 64");
    }
    if (cfg.parallel_sessions <= 0 || cfg.parallel_sessions > 64) {
        throw std::invalid_argument("--parallel-sessions must be between 1 and 64");
    }
    if (cfg.decode_batch_size <= 0 || cfg.decode_batch_size > cfg.parallel_sessions) {
        throw std::invalid_argument(
            "--decode-batch-size must be between 1 and --parallel-sessions"
        );
    }
    if (cfg.decode_batch_size > cfg.http_workers) {
        throw std::invalid_argument(
            "--decode-batch-size cannot exceed --http-workers"
        );
    }
    if (cfg.speculative_draft != "none" && cfg.speculative_draft != "prompt_lookup") {
        throw std::invalid_argument("--speculative-draft must be none or prompt_lookup");
    }
    if (cfg.speculative_max_tokens <= 0 || cfg.speculative_max_tokens > 8) {
        throw std::invalid_argument("--speculative-max-tokens must be between 1 and 8");
    }
    if (cfg.speculative_draft != "none" && cfg.decode_batch_size != 1) {
        throw std::invalid_argument(
            "speculative decoding currently requires --decode-batch-size 1"
        );
    }
    if (cfg.speculative_draft != "none" && cfg.engine != "llama.cpp") {
        throw std::invalid_argument(
            "speculative decoding requires an engine with multi-token verification and KV rollback"
        );
    }
    if (cfg.mode == ExecutionMode::TensorParallel) {
        if (cfg.engine != "llama.cpp") {
            throw std::invalid_argument("tensor_parallel currently requires --engine llama.cpp");
        }
        if (cfg.compute_backend != "cuda") {
            throw std::invalid_argument("tensor_parallel requires --compute-backend cuda");
        }
        if (cfg.kv_cache_type != KVCacheType::F16) {
            throw std::invalid_argument("tensor_parallel currently requires --kv-cache-type f16");
        }
        const std::string mesh_error = tensor_parallel::validate_device_mesh(cfg.tensor_mesh);
        if (!mesh_error.empty()) {
            throw std::invalid_argument(mesh_error);
        }
    } else if (!cfg.tensor_mesh.remote_endpoints.empty() ||
               !cfg.tensor_mesh.tensor_split.empty()) {
        throw std::invalid_argument(
            "tensor RPC peers and tensor split require --mode tensor_parallel"
        );
    }
}

} // namespace

void validate_deployment_config(const Config& cfg) {
    if (cfg.node_name.empty()) {
        throw std::invalid_argument("--node-name must not be empty");
    }
    if (cfg.model.empty()) {
        throw std::invalid_argument("--model must not be empty");
    }
    validate_engine_config(cfg);
    if (cfg.engine == "native") {
        if (cfg.mode != ExecutionMode::PipelineParallel) {
            throw std::invalid_argument(
                "native serving currently requires --mode pipeline_parallel"
            );
        }
        if (cfg.stage_assignment.stage_index != 0 ||
            cfg.stage_assignment.stage_count != 1) {
            throw std::invalid_argument(
                "native serving currently requires one logical stage at index zero"
            );
        }
        if (cfg.compute_backend != "cpu" && cfg.compute_backend != "cuda") {
            throw std::invalid_argument("native compute backend must be cpu or cuda");
        }
        if (cfg.decode_batch_size != 1) {
            throw std::invalid_argument(
                "native serving currently requires --decode-batch-size 1"
            );
        }
        if (cfg.kv_cache_type != KVCacheType::F16) {
            throw std::invalid_argument("native serving currently requires --kv-cache-type f16");
        }
        if (cfg.speculative_draft != "none") {
            throw std::invalid_argument(
                "native serving does not yet support speculative decoding"
            );
        }
    }
    if (cfg.mode == ExecutionMode::PipelineParallel &&
        cfg.stage_assignment.layer_end <= cfg.stage_assignment.layer_start) {
        throw std::invalid_argument("pipeline_parallel mode requires --layer-end greater than --layer-start");
    }
    if (cfg.mode != ExecutionMode::PipelineParallel && cfg.stage_assignment.stage_count > 1) {
        throw std::invalid_argument("multi-stage assignment requires --mode pipeline_parallel");
    }
    if (cfg.mode == ExecutionMode::TensorParallel &&
        (cfg.stage_assignment.stage_index != 0 || cfg.stage_assignment.stage_count != 1)) {
        throw std::invalid_argument("tensor_parallel requires one logical stage at index zero");
    }
    const std::string stage_error = pipeline_parallel::validate_stage_assignment(cfg.stage_assignment);
    if (!stage_error.empty()) {
        throw std::invalid_argument("invalid stage assignment: " + stage_error);
    }
}

void validate_runtime_config(const Config& cfg) {
    if (cfg.host.empty()) {
        throw std::invalid_argument("listen host must not be empty");
    }
    if (cfg.port <= 0 || cfg.port > 65535) {
        throw std::invalid_argument("listen port must be between 1 and 65535");
    }
    if (cfg.node_name.empty()) {
        throw std::invalid_argument("--node-name must not be empty");
    }
    if (cfg.cluster_token.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument("JETSONFABRIC_CLUSTER_TOKEN must not contain newlines");
    }
    if (!is_loopback_host(cfg.host) && cfg.cluster_token.empty()) {
        throw std::invalid_argument(
            "non-loopback --listen requires JETSONFABRIC_CLUSTER_TOKEN"
        );
    }
    validate_engine_config(cfg);
    if (!cfg.start_idle) {
        validate_deployment_config(cfg);
    }
}

void print_help() {
    std::cout
        << "jetsonfabric-runtime-worker\n\n"
        << "Flags:\n"
        << "  --listen host:port       listen address, default 127.0.0.1:9090\n"
        << "  --node-name name         logical node name for this worker\n"
        << "  --idle                   start without loading a resident deployment\n"
        << "  --model model-id         model id served by configured startup\n"
        << "  --mode mode              data_parallel, pipeline_parallel, tensor_parallel\n"
        << "  --stage-index n          stage index, zero-based\n"
        << "  --stage-count n          total number of ordered stages\n"
        << "  --layer-start n          first transformer layer, inclusive\n"
        << "  --layer-end n            transformer layer end, exclusive\n"
        << "  --tensor-transport n     tensor transport, currently llama_rpc\n"
        << "  --tensor-rpc-peers n     comma-separated remote GGML RPC host:port endpoints\n"
        << "  --tensor-split n         optional local,remote device weight proportions\n"
        << "  --engine engine          registered inference engine name\n"
        << "  --stage-transport name   registered peer-stage transport name\n"
        << "  --activation-encoding n  inter-stage activation encoding: f32 or f16\n"
        << "  --compute-backend name   compute backend passed to the engine\n"
        << "  --model-path path        GGUF model path for llama.cpp\n"
        << "  --ctx-size n             context size, default 4096\n"
        << "  --ubatch-size n          llama.cpp physical micro-batch size, default 512\n"
        << "  --kv-cache-type type     llama.cpp KV cache type: f16 or q8_0\n"
        << "  --n-gpu-layers n         llama.cpp GPU layers, default 999\n"
        << "  --threads n              CPU threads, default 0\n"
        << "  --http-workers n         bounded HTTP worker count, default 2\n"
        << "  --parallel-sessions n    resident session capacity when batching, default 2\n"
        << "  --decode-batch-size n    continuous decode batch size, default 1 (disabled)\n"
        << "  --speculative-draft n    none or prompt_lookup, default none\n"
        << "  --speculative-max-tokens maximum drafted tokens per target pass, default 4\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    if (const char* token = std::getenv("JETSONFABRIC_CLUSTER_TOKEN"); token != nullptr) {
        cfg.cluster_token = token;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--listen") {
            parse_listen(cfg, require_value(i, argc, argv, arg));
        } else if (arg == "--node-name") {
            cfg.node_name = require_value(i, argc, argv, arg);
        } else if (arg == "--idle") {
            cfg.start_idle = true;
        } else if (arg == "--model") {
            cfg.model = require_value(i, argc, argv, arg);
        } else if (arg == "--mode") {
            const std::string value = require_value(i, argc, argv, arg);
            try {
                cfg.mode = parse_execution_mode(value);
            } catch (const std::invalid_argument& err) {
                fail(err.what());
            }
        } else if (arg == "--stage-index") {
            cfg.stage_assignment.stage_index = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--stage-count") {
            cfg.stage_assignment.stage_count = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--layer-start") {
            cfg.stage_assignment.layer_start = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--layer-end") {
            cfg.stage_assignment.layer_end = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--tensor-transport") {
            cfg.tensor_mesh.transport = require_value(i, argc, argv, arg);
        } else if (arg == "--tensor-rpc-peers") {
            cfg.tensor_mesh.remote_endpoints = tensor_parallel::parse_remote_endpoints(
                require_value(i, argc, argv, arg)
            );
        } else if (arg == "--tensor-split") {
            try {
                cfg.tensor_mesh.tensor_split = tensor_parallel::parse_tensor_split(
                    require_value(i, argc, argv, arg)
                );
            } catch (const std::exception& error) {
                fail(std::string("invalid --tensor-split: ") + error.what());
            }
        } else if (arg == "--engine") {
            cfg.engine = require_value(i, argc, argv, arg);
        } else if (arg == "--stage-transport") {
            cfg.stage_transport = require_value(i, argc, argv, arg);
        } else if (arg == "--activation-encoding") {
            cfg.activation_encoding = require_value(i, argc, argv, arg);
        } else if (arg == "--compute-backend") {
            cfg.compute_backend = require_value(i, argc, argv, arg);
        } else if (arg == "--model-path") {
            cfg.model_path = require_value(i, argc, argv, arg);
        } else if (arg == "--ctx-size") {
            cfg.ctx_size = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--ubatch-size") {
            cfg.ubatch_size = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--kv-cache-type") {
            const std::string value = require_value(i, argc, argv, arg);
            try {
                cfg.kv_cache_type = parse_kv_cache_type(value);
            } catch (const std::invalid_argument& err) {
                fail(err.what());
            }
        } else if (arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--threads") {
            cfg.threads = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--http-workers") {
            cfg.http_workers = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--parallel-sessions") {
            cfg.parallel_sessions = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--decode-batch-size") {
            cfg.decode_batch_size = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--speculative-draft") {
            cfg.speculative_draft = require_value(i, argc, argv, arg);
        } else if (arg == "--speculative-max-tokens") {
            cfg.speculative_max_tokens = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            fail("unknown arg: " + arg);
        }
    }

    try {
        validate_runtime_config(cfg);
    } catch (const std::invalid_argument& err) {
        fail(err.what());
    }

    std::cerr
        << "runtime configuration: state=" << (cfg.start_idle ? "idle" : "active")
        << " engine=" << cfg.engine
        << " stage_transport=" << cfg.stage_transport
        << " activation_encoding=" << cfg.activation_encoding
        << " compute_backend=" << cfg.compute_backend
        << " n_gpu_layers=" << cfg.n_gpu_layers
        << " ctx_size=" << cfg.ctx_size
        << " ubatch_size=" << cfg.ubatch_size
        << " kv_cache_type=" << kv_cache_type_string(cfg.kv_cache_type)
        << " http_workers=" << cfg.http_workers
        << " parallel_sessions=" << cfg.parallel_sessions
        << " decode_batch_size=" << cfg.decode_batch_size
        << " speculative_draft=" << cfg.speculative_draft
        << " speculative_max_tokens=" << cfg.speculative_max_tokens;
    if (cfg.mode == ExecutionMode::TensorParallel) {
        std::cerr
            << " tensor_transport=" << cfg.tensor_mesh.transport
            << " tensor_world_size=" << cfg.tensor_mesh.world_size();
    }
    if (!cfg.start_idle) {
        std::cerr
            << " stage=" << cfg.stage_assignment.stage_index << "/" << cfg.stage_assignment.stage_count
            << " layers=[" << cfg.stage_assignment.layer_start << "," << cfg.stage_assignment.layer_end << ")";
    }
    std::cerr << "\n";
    return cfg;
}

} // namespace jetsonfabric::runtime
