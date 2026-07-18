#include "worker/config.hpp"

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

void parse_listen(Config& cfg, const std::string& value) {
    const auto colon = value.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) fail("--listen must be host:port");
    cfg.host = value.substr(0, colon);
    cfg.port = parse_int(value.substr(colon + 1), "--listen port");
    if (cfg.port <= 0 || cfg.port > 65535) fail("--listen port must be between 1 and 65535");
}

void validate_config(const Config& cfg) {
    if (cfg.host.empty()) fail("listen host must not be empty");
    if (cfg.port <= 0 || cfg.port > 65535) fail("listen port must be between 1 and 65535");
    if (cfg.node_name.empty()) fail("--node-name must not be empty");
    if (cfg.model.empty()) fail("--model must not be empty");
    if (cfg.mode == ExecutionMode::PipelineParallel && cfg.stage_assignment.layer_end <= cfg.stage_assignment.layer_start) {
        fail("pipeline_parallel mode requires --layer-end greater than --layer-start");
    }
    if (cfg.mode != ExecutionMode::PipelineParallel && cfg.stage_assignment.stage_count > 1) {
        fail("multi-stage assignment requires --mode pipeline_parallel");
    }
    const std::string stage_error = pipeline_parallel::validate_stage_assignment(cfg.stage_assignment);
    if (!stage_error.empty()) fail("invalid stage assignment: " + stage_error);
    if (cfg.engine != "llama.cpp" && cfg.engine != "synthetic") {
        fail("--engine supports llama.cpp or synthetic");
    }
    if (cfg.engine == "llama.cpp" && cfg.model_path.empty()) {
        fail("--model-path is required when --engine llama.cpp");
    }
    if (cfg.compute_backend != "cpu" && cfg.compute_backend != "cuda") {
        fail("--compute-backend must be cpu or cuda");
    }
}

} // namespace

void print_help() {
    std::cout
        << "jetsonfabric-runtime-worker\n\n"
        << "Flags:\n"
        << "  --listen host:port       listen address, default 127.0.0.1:9090\n"
        << "  --node-name name         logical node name for this worker\n"
        << "  --model model-id         model id served by this runtime\n"
        << "  --mode mode              data_parallel, pipeline_parallel, tensor_parallel\n"
        << "  --stage-index n          stage index, zero-based\n"
        << "  --stage-count n          total number of ordered stages\n"
        << "  --layer-start n          first transformer layer, inclusive\n"
        << "  --layer-end n            transformer layer end, exclusive\n"
        << "  --engine engine          hosted inference engine: llama.cpp or synthetic\n"
        << "  --compute-backend name   local compute backend: cpu or cuda\n"
        << "  --model-path path        GGUF model path for llama.cpp\n"
        << "  --ctx-size n             context size, default 4096\n"
        << "  --n-gpu-layers n         llama.cpp GPU layers, default 999\n"
        << "  --threads n              CPU threads, default 0\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--listen") {
            parse_listen(cfg, require_value(i, argc, argv, arg));
        } else if (arg == "--node-name") {
            cfg.node_name = require_value(i, argc, argv, arg);
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
        } else if (arg == "--engine") {
            cfg.engine = require_value(i, argc, argv, arg);
        } else if (arg == "--compute-backend") {
            cfg.compute_backend = require_value(i, argc, argv, arg);
        } else if (arg == "--model-path") {
            cfg.model_path = require_value(i, argc, argv, arg);
        } else if (arg == "--ctx-size") {
            cfg.ctx_size = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--threads") {
            cfg.threads = parse_int(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            fail("unknown arg: " + arg);
        }
    }

    if (cfg.ctx_size <= 0) fail("--ctx-size must be greater than zero");
    if (cfg.n_gpu_layers < 0) fail("--n-gpu-layers must be zero or greater");
    if (cfg.threads < 0) fail("--threads must be zero or greater");

    validate_config(cfg);
    std::cerr
        << "runtime configuration: engine=" << cfg.engine
        << " compute_backend=" << cfg.compute_backend
        << " n_gpu_layers=" << cfg.n_gpu_layers
        << " ctx_size=" << cfg.ctx_size
        << " stage=" << cfg.stage_assignment.stage_index << "/" << cfg.stage_assignment.stage_count
        << " layers=[" << cfg.stage_assignment.layer_start << "," << cfg.stage_assignment.layer_end << ")\n";
    return cfg;
}

} // namespace jetsonfabric::runtime
