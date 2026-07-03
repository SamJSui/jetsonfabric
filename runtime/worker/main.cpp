#include "api/http_server.hpp"
#include "engine/stub_engine.hpp"
#include "worker/config.hpp"

#include <atomic>
#include <csignal>
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

    jetsonfabric::runtime::Config config = jetsonfabric::runtime::parse_args(argc, argv);
    jetsonfabric::runtime::StubEngine engine(config.model, config.mode);
    jetsonfabric::runtime::HttpServer server(config, engine, g_running);

    return server.run();
}
