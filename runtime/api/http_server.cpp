#include "api/http_server.hpp"

#include "api/http_response.hpp"
#include "protocol/execution_mode.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::runtime {
namespace {

constexpr std::size_t kMaxHeaderBytes = 64U << 10;
constexpr std::size_t kMaxBodyBytes = (512U << 20) + (1U << 20) + 20U;

std::string health_body(const RuntimeAPI& runtime) {
    std::ostringstream body;
    body << "{"
         << "\"status\":\"ok\","
         << "\"runtime\":\"" << runtime.runtime_name() << "\","
         << "\"engine\":\"" << runtime.engine_name() << "\","
         << "\"mode\":\"" << execution_mode_string(runtime.execution_mode()) << "\","
         << "\"model\":\"" << runtime.model() << "\""
         << "}";
    return body.str();
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::optional<std::size_t> content_length(const std::string& headers) {
    std::istringstream lines(headers);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (lower(line.substr(0, colon)) != "content-length") continue;
        std::string value = line.substr(colon + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size() || parsed > kMaxBodyBytes) {
            throw std::invalid_argument("invalid or oversized Content-Length");
        }
        return static_cast<std::size_t>(parsed);
    }
    return std::nullopt;
}

std::string read_http_request(int client_fd) {
    std::string request;
    request.reserve(8192);
    std::size_t header_end = std::string::npos;
    std::optional<std::size_t> body_length;
    char buffer[8192];

    while (true) {
        const ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
        }
        if (n == 0) break;
        request.append(buffer, static_cast<std::size_t>(n));

        if (header_end == std::string::npos) {
            const std::size_t marker = request.find("\r\n\r\n");
            if (marker != std::string::npos) {
                header_end = marker + 4;
                if (header_end > kMaxHeaderBytes) {
                    throw std::invalid_argument("HTTP headers are too large");
                }
                body_length = content_length(request.substr(0, marker));
            } else if (request.size() > kMaxHeaderBytes) {
                throw std::invalid_argument("HTTP headers are too large");
            }
        }

        if (header_end != std::string::npos) {
            const std::size_t expected_body = body_length.value_or(0);
            if (request.size() >= header_end + expected_body) {
                request.resize(header_end + expected_body);
                return request;
            }
        }
        if (request.size() > kMaxHeaderBytes + kMaxBodyBytes) {
            throw std::invalid_argument("HTTP request body is too large");
        }
    }

    if (header_end == std::string::npos) {
        throw std::invalid_argument("incomplete HTTP request headers");
    }
    const std::size_t expected_body = body_length.value_or(0);
    if (request.size() != header_end + expected_body) {
        throw std::invalid_argument("truncated HTTP request body");
    }
    return request;
}

std::string request_body(const std::string& request) {
    const std::size_t marker = request.find("\r\n\r\n");
    return marker == std::string::npos ? std::string{} : request.substr(marker + 4);
}

void send_all(int fd, const std::string& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t sent = send(fd, data.data() + offset, data.size() - offset, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return;
        }
        if (sent == 0) return;
        offset += static_cast<std::size_t>(sent);
    }
}

} // namespace

HttpServer::HttpServer(Config config, const RuntimeAPI& runtime, std::atomic_bool& running)
    : config_(std::move(config)), runtime_(runtime), running_(running) {}

int HttpServer::run() const {
    int server_fd = open_listening_socket();
    if (server_fd < 0) return 1;

    std::cout << runtime_.runtime_name() << " listening on http://"
              << config_.host << ":" << config_.port
              << " engine=" << runtime_.engine_name()
              << " model=" << config_.model
              << " mode=" << execution_mode_string(runtime_.execution_mode()) << "\n";

    while (running_.load()) {
        if (!wait_for_client(server_fd)) continue;
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            if (running_.load()) std::cerr << "accept failed: " << std::strerror(errno) << "\n";
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
        std::cerr << "bind failed on " << config_.host << ":" << config_.port << ": " << std::strerror(errno) << "\n";
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
    timeout.tv_usec = 250000;
    const int ready = select(server_fd + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready < 0) {
        if (errno != EINTR) std::cerr << "select failed: " << std::strerror(errno) << "\n";
        return false;
    }
    return ready > 0 && FD_ISSET(server_fd, &read_fds);
}

void HttpServer::handle_client(int client_fd) const {
    HttpResponse response;
    try {
        const std::string request = read_http_request(client_fd);
        const std::string body = request_body(request);
        response = not_found_response();
        if (starts_with(request, "GET /healthz ")) {
            response = json_response("200 OK", health_body(runtime_));
        } else if (starts_with(request, "POST /v1/chat/completions ")) {
            const RuntimeResponse runtime_response = runtime_.chat_completion(body);
            response = binary_response(runtime_response.status, runtime_response.content_type, runtime_response.body);
        } else if (starts_with(request, "POST /v1/layer-split/stage ")) {
            const RuntimeResponse runtime_response = runtime_.run_stage(body);
            response = binary_response(runtime_response.status, runtime_response.content_type, runtime_response.body);
        }
    } catch (const std::exception& err) {
        response = json_response("400 Bad Request", std::string("{\"error\":\"invalid_http_request\",\"message\":\"") + err.what() + "\"}");
    }
    send_all(client_fd, response.serialize());
    close(client_fd);
}

} // namespace jetsonfabric::runtime
