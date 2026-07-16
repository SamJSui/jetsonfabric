#include "api/http_server.hpp"
#include "engine/runtime_service.hpp"
#include "worker/config.hpp"

#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>

namespace {

std::atomic_bool g_running{true};

void handle_signal(int) {
    g_running.store(false);
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try {
        jetsonfabric::runtime::Config config = jetsonfabric::runtime::parse_args(argc, argv);
        jetsonfabric::runtime::RuntimeService runtime(config);
        jetsonfabric::runtime::HttpServer server(config, runtime, g_running);
        return server.run();
    } catch (const std::exception& err) {
        std::cerr << "runtime startup failed: " << err.what() << "\n";
        return 1;
    }
}
