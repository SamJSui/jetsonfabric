#include "worker/config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace jetsonfabric::runtime {
namespace {

std::string require_value(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        std::exit(2);
    }
    return argv[++index];
}

void parse_listen(Config& cfg, const std::string& value) {
    auto colon = value.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) {
        std::cerr << "--listen must be host:port\n";
        std::exit(2);
    }
    cfg.host = value.substr(0, colon);
    cfg.port = std::stoi(value.substr(colon + 1));
    if (cfg.port <= 0 || cfg.port > 65535) {
        std::cerr << "--listen port must be between 1 and 65535\n";
        std::exit(2);
    }
}

} // namespace

void print_help() {
    std::cout
        << "jetsonfabric-runtime-worker\n\n"
        << "Flags:\n"
        << "  --listen host:port   listen address, default 127.0.0.1:9090\n"
        << "  --model model-id     model id to report in stub responses\n"
        << "  --mode mode          runtime mode, default single_node\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--listen") {
            parse_listen(cfg, require_value(i, argc, argv, arg));
        } else if (arg == "--model") {
            cfg.model = require_value(i, argc, argv, arg);
        } else if (arg == "--mode") {
            cfg.mode = require_value(i, argc, argv, arg);
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << arg << "\n";
            std::exit(2);
        }
    }

    return cfg;
}

} // namespace jetsonfabric::runtime
