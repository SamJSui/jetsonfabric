#pragma once

#include <string>

namespace jetsonfabric::runtime {

struct Config {
    std::string host = "127.0.0.1";
    int port = 9090;
    std::string model = "runtime-stub-model";
    std::string mode = "single_node";
};

Config parse_args(int argc, char** argv);
void print_help();

} // namespace jetsonfabric::runtime
