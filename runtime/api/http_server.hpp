#pragma once

#include "engine/engine.hpp"
#include "worker/config.hpp"

#include <atomic>

namespace jetsonfabric::runtime {

class HttpServer {
public:
    HttpServer(Config config, const Engine& engine, std::atomic_bool& running);

    int run() const;

private:
    Config config_;
    const Engine& engine_;
    std::atomic_bool& running_;

    void handle_client(int client_fd) const;
};

} // namespace jetsonfabric::runtime
