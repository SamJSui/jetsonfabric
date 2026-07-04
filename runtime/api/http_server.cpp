#include "api/http_server.hpp"

#include "api/http_response.hpp"
#include "protocol/execution_mode.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
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
        << "\"mode\":\"" << execution_mode_string(engine.mode()) << "\","
        << "\"model\":\"" << engine.model() << "\""
        << "}";
    return body.str();
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string request_body(const std::string& request) {
    const std::string marker = "\r\n\r\n";
    const std::size_t pos = request.find(marker);
    if (pos == std::string::npos) {
        return "";
    }
    return request.substr(pos + marker.size());
}

} // namespace

HttpServer::HttpServer(Config config, const Engine& engine, std::atomic_bool& running)
    : config_(std::move(config)), engine_(engine), running_(running) {}

int HttpServer::run() const {
    int server_fd = open_listening_socket();
    if (server_fd < 0) {
        return 1;
    }

    std::cout
        << "jetsonfabric-runtime-worker listening on http://"
        << config_.host << ":" << config_.port
        << " model=" << config_.model
        << " mode=" << execution_mode_string(engine_.mode())
        << "\n";

    while (running_.load()) {
        if (!wait_for_client(server_fd)) {
            continue;
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }

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

int HttpServer::open_listening_socket() const {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << "\n";
        return -1;
    }

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "setsockopt failed: " << std::strerror(errno) << "\n";
        close(server_fd);
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));

    if (inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << "invalid host: " << config_.host << "\n";
        close(server_fd);
        return -1;
    }

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed on " << config_.host << ":" << config_.port << ": "
                  << std::strerror(errno) << "\n";
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 16) < 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << "\n";
        close(server_fd);
        return -1;
    }

    return server_fd;
}

bool HttpServer::wait_for_client(int server_fd) const {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(server_fd, &read_fds);

    timeval timeout{};
    timeout.tv_sec = 0;
    timeout.tv_usec = 250000;

    int ready = select(server_fd + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready < 0) {
        if (errno == EINTR) {
            return false;
        }

        std::cerr << "select failed: " << std::strerror(errno) << "\n";
        return false;
    }

    return ready > 0 && FD_ISSET(server_fd, &read_fds);
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
    const std::string body = request_body(request);

    HttpResponse response = not_found_response();

    if (starts_with(request, "GET /healthz ")) {
        response = json_response("200 OK", health_body(engine_));
    } else if (starts_with(request, "POST /v1/chat/completions ")) {
        const EngineResponse engine_response = engine_.chat_completion(body);
        response = json_response(engine_response.status, engine_response.body);
    } else if (starts_with(request, "POST /v1/layer-split/stage ")) {
        const EngineResponse engine_response = engine_.run_stage(body);
        response = json_response(engine_response.status, engine_response.body);
    }

    std::string serialized = response.serialize();
    send(client_fd, serialized.data(), serialized.size(), 0);
    close(client_fd);
}

} // namespace jetsonfabric::runtime