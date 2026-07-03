#include "api/http_server.hpp"

#include "api/http_response.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace jetsonfabric::runtime {
namespace {

std::string health_body(const Engine& engine) {
    std::ostringstream body;
    body
        << "{"
        << "\"status\":\"ok\","
        << "\"runtime\":\"" << engine.runtime_name() << "\","
        << "\"mode\":\"" << engine.mode() << "\","
        << "\"model\":\"" << engine.model() << "\""
        << "}";
    return body.str();
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

} // namespace

HttpServer::HttpServer(Config config, const Engine& engine, std::atomic_bool& running)
    : config_(std::move(config)), engine_(engine), running_(running) {}

int HttpServer::run() const {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "setsockopt failed: " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));

    if (inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << "invalid host: " << config_.host << "\n";
        close(server_fd);
        return 1;
    }

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed on " << config_.host << ":" << config_.port << ": "
                  << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 16) < 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << "\n";
        close(server_fd);
        return 1;
    }

    std::cout
        << "jetsonfabric-runtime-worker listening on http://"
        << config_.host << ":" << config_.port
        << " model=" << config_.model
        << " mode=" << config_.mode
        << "\n";

    while (running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

        if (client_fd < 0) {
            if (running_.load()) {
                std::cerr << "accept failed: " << std::strerror(errno) << "\n";
            }
            continue;
        }

        handle_client(client_fd);
    }

    close(server_fd);
    return 0;
}

void HttpServer::handle_client(int client_fd) const {
    char buffer[8192];
    std::memset(buffer, 0, sizeof(buffer));

    ssize_t n = read(client_fd, buffer, sizeof(buffer) - 1);
    if (n <= 0) {
        close(client_fd);
        return;
    }

    std::string request(buffer, static_cast<size_t>(n));
    HttpResponse response = not_found_response();

    if (starts_with(request, "GET /healthz ")) {
        response = json_response("200 OK", health_body(engine_));
    } else if (starts_with(request, "POST /v1/chat/completions ")) {
        response = json_response("200 OK", engine_.chat_completion(request));
    }

    std::string serialized = response.serialize();
    send(client_fd, serialized.data(), serialized.size(), 0);
    close(client_fd);
}

} // namespace jetsonfabric::runtime
