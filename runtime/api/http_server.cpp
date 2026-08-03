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
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime {
namespace {

constexpr std::size_t kMaxHeaderBytes = 64U << 10;
constexpr std::size_t kMaxBodyBytes = (512U << 20) + (1U << 20) + 20U;
constexpr int kStageKeepAliveIdleSeconds = 5;
constexpr int kHTTPIOTimeoutSeconds = 30;

std::string health_body(const RuntimeAPI& runtime) {
    std::ostringstream body;
    body << "{"
         << "\"status\":\"ok\","
         << "\"runtime\":\"" << runtime.runtime_name() << "\","
         << "\"engine\":\"" << runtime.engine_name() << "\","
         << "\"mode\":\"" << execution_mode_string(runtime.execution_mode()) << "\","
         << "\"stage_transport\":\"" << runtime.stage_transport_name() << "\","
         << "\"activation_encoding\":\"" << runtime.activation_encoding() << "\","
         << "\"kv_cache_type\":\"" << runtime.kv_cache_type() << "\","
         << "\"ubatch_size\":" << runtime.ubatch_size() << ","
         << "\"parallel_sessions\":" << runtime.parallel_sessions() << ","
         << "\"decode_batch_size\":" << runtime.decode_batch_size() << ","
         << "\"speculative_draft\":\"" << runtime.speculative_draft() << "\","
         << "\"speculative_max_tokens\":" << runtime.speculative_max_tokens() << ","
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

std::optional<std::string> header_value(
    const std::string& request,
    const std::string& name
) {
    const std::size_t header_end = request.find("\r\n\r\n");
    if (header_end == std::string::npos) return std::nullopt;
    std::istringstream lines(request.substr(0, header_end));
    std::string line;
    const std::string normalized_name = lower(name);
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos || lower(line.substr(0, colon)) != normalized_name) {
            continue;
        }
        std::string value = line.substr(colon + 1);
        const std::size_t first = value.find_first_not_of(" \t");
        const std::size_t last = value.find_last_not_of(" \t");
        return first == std::string::npos
            ? std::string{}
            : value.substr(first, last - first + 1);
    }
    return std::nullopt;
}

bool constant_time_equal(const std::string& left, const std::string& right) {
    std::size_t difference = left.size() ^ right.size();
    const std::size_t length = std::max(left.size(), right.size());
    for (std::size_t index = 0; index < length; ++index) {
        const unsigned char left_value = index < left.size()
            ? static_cast<unsigned char>(left[index])
            : 0;
        const unsigned char right_value = index < right.size()
            ? static_cast<unsigned char>(right[index])
            : 0;
        difference |= left_value ^ right_value;
    }
    return difference == 0;
}

bool runtime_write_authorized(const std::string& request, const Config& config) {
    if (config.cluster_token.empty()) return true;
    const auto provided = header_value(request, "x-jetsonfabric-cluster-token");
    return provided.has_value() && constant_time_equal(*provided, config.cluster_token);
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

std::optional<std::string> read_http_request(int client_fd, std::string& buffered) {
    char buffer[8192];

    while (true) {
        const std::size_t marker = buffered.find("\r\n\r\n");
        if (marker != std::string::npos) {
            const std::size_t header_end = marker + 4;
            if (header_end > kMaxHeaderBytes) {
                throw std::invalid_argument("HTTP headers are too large");
            }
            const std::size_t expected_body =
                content_length(buffered.substr(0, marker)).value_or(0);
            const std::size_t request_size = header_end + expected_body;
            if (buffered.size() >= request_size) {
                std::string request = buffered.substr(0, request_size);
                buffered.erase(0, request_size);
                return request;
            }
        } else if (buffered.size() > kMaxHeaderBytes) {
            throw std::invalid_argument("HTTP headers are too large");
        }
        if (buffered.size() > kMaxHeaderBytes + kMaxBodyBytes) {
            throw std::invalid_argument("HTTP request body is too large");
        }

        const ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            if ((errno == EAGAIN || errno == EWOULDBLOCK) && buffered.empty()) {
                return std::nullopt;
            }
            throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            if (buffered.empty()) return std::nullopt;
            break;
        }
        buffered.append(buffer, static_cast<std::size_t>(n));
    }

    const std::size_t marker = buffered.find("\r\n\r\n");
    if (marker == std::string::npos) {
        throw std::invalid_argument("incomplete HTTP request headers");
    }
    const std::size_t header_end = marker + 4;
    const std::size_t expected_body =
        content_length(buffered.substr(0, marker)).value_or(0);
    if (buffered.size() < header_end + expected_body) {
        throw std::invalid_argument("truncated HTTP request body");
    }
    std::string request = buffered.substr(0, header_end + expected_body);
    buffered.erase(0, header_end + expected_body);
    return request;
}

std::string request_body(const std::string& request) {
    const std::size_t marker = request.find("\r\n\r\n");
    return marker == std::string::npos ? std::string{} : request.substr(marker + 4);
}

bool requests_stage_keep_alive(const std::string& request) {
    if (!starts_with(request, "POST /v1/layer-split/stage ")) return false;
    const std::size_t header_end = request.find("\r\n\r\n");
    if (header_end == std::string::npos) return false;
    const std::string headers = lower(request.substr(0, header_end) + "\r\n");
    return headers.find("\r\nconnection: keep-alive\r\n") != std::string::npos;
}

void set_stage_keep_alive_timeout(int client_fd) {
    timeval timeout{.tv_sec = kStageKeepAliveIdleSeconds, .tv_usec = 0};
    (void) setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
}

void set_http_io_timeout(int client_fd) {
    timeval timeout{.tv_sec = kHTTPIOTimeoutSeconds, .tv_usec = 0};
    (void) setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void) setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool send_all(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t sent = send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        offset += static_cast<std::size_t>(sent);
    }
    return true;
}

bool send_all(int fd, const std::string& data) {
    return send_all(fd, data.data(), data.size());
}

bool send_all(int fd, std::span<const std::uint8_t> data) {
    return send_all(fd, data.data(), data.size());
}

bool send_response(int fd, const HttpResponse& response) {
    return send_all(fd, response.serialize_headers()) &&
        send_all(fd, response.body) &&
        send_all(fd, response.body_payload);
}

HttpResponse runtime_http_response(RuntimeResponse response) {
    return binary_response(
        std::move(response.status),
        std::move(response.content_type),
        std::move(response.body),
        std::move(response.body_payload)
    );
}

bool send_chunk(int fd, const std::string& data) {
    std::ostringstream chunk;
    chunk << std::hex << data.size() << "\r\n";
    chunk << data << "\r\n";
    return send_all(fd, chunk.str());
}

bool send_generation_headers(int fd) {
    return send_all(
        fd,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/x-ndjson\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n"
    );
}

} // namespace

HttpServer::HttpServer(Config config, RuntimeAPI& runtime, std::atomic_bool& running)
    : config_(std::move(config)), runtime_(runtime), running_(running) {}

int HttpServer::run() const {
    int server_fd = open_listening_socket();
    if (server_fd < 0) return 1;

    std::cout << runtime_.runtime_name() << " listening on http://"
              << config_.host << ":" << config_.port
              << " engine=" << runtime_.engine_name()
              << " model=" << runtime_.model()
              << " mode=" << execution_mode_string(runtime_.execution_mode()) << "\n";

    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::deque<int> clients;
    std::set<int> active_clients;
    bool accepting = true;
    const std::size_t queue_capacity =
        static_cast<std::size_t>(config_.http_workers) * 8;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(config_.http_workers));
    for (int index = 0; index < config_.http_workers; ++index) {
        workers.emplace_back([this, &queue_mutex, &queue_ready, &clients, &active_clients, &accepting]() {
            while (true) {
                int client_fd = -1;
                {
                    std::unique_lock lock(queue_mutex);
                    queue_ready.wait(lock, [&clients, &accepting]() {
                        return !clients.empty() || !accepting;
                    });
                    if (clients.empty()) {
                        return;
                    }
                    client_fd = clients.front();
                    clients.pop_front();
                    active_clients.insert(client_fd);
                }
                handle_client(client_fd);
                {
                    const std::lock_guard lock(queue_mutex);
                    active_clients.erase(client_fd);
                }
                close(client_fd);
            }
        });
    }

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
        set_http_io_timeout(client_fd);
        bool queued = false;
        {
            const std::lock_guard lock(queue_mutex);
            if (clients.size() < queue_capacity) {
                clients.push_back(client_fd);
                queued = true;
            }
        }
        if (queued) {
            queue_ready.notify_one();
        } else {
            (void) send_all(
                client_fd,
                json_response(
                    "503 Service Unavailable",
                    "{\"error\":\"runtime_http_queue_full\"}"
                ).serialize()
            );
            close(client_fd);
        }
    }
    close(server_fd);
    runtime_.shutdown();
    {
        const std::lock_guard lock(queue_mutex);
        accepting = false;
        for (int client_fd : clients) {
            (void) shutdown(client_fd, SHUT_RDWR);
            close(client_fd);
        }
        clients.clear();
        for (int client_fd : active_clients) {
            (void) shutdown(client_fd, SHUT_RDWR);
        }
    }
    queue_ready.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
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
    std::string buffered;
    buffered.reserve(8192);
    while (true) {
        HttpResponse response;
        bool keep_alive = false;
        try {
            const std::optional<std::string> wire_request =
                read_http_request(client_fd, buffered);
            if (!wire_request.has_value()) break;
            const std::string& request = *wire_request;
            const std::string body = request_body(request);
            if (starts_with(request, "POST ") && !runtime_write_authorized(request, config_)) {
                response = json_response(
                    "401 Unauthorized",
                    "{\"error\":\"runtime_auth_failed\",\"message\":\"runtime writes require the configured cluster token\"}"
                );
            } else if (starts_with(request, "POST /v1/generate ")) {
                if (!send_generation_headers(client_fd)) break;
                const RuntimeResponse final_event = runtime_.generate(
                    body,
                    [client_fd](const std::string& event) {
                        return send_chunk(client_fd, event + "\n");
                    }
                );
                if (!final_event.body.empty()) {
                    (void) send_chunk(client_fd, final_event.body + "\n");
                }
                (void) send_all(client_fd, "0\r\n\r\n");
                break;
            } else {
                response = not_found_response();
                if (starts_with(request, "GET /healthz ")) {
                    response = json_response("200 OK", health_body(runtime_));
                } else if (starts_with(request, "GET /v1/deployment ")) {
                    response = runtime_http_response(runtime_.deployment_status());
                } else if (starts_with(request, "POST /v1/deployment/load ")) {
                    response = runtime_http_response(runtime_.load_deployment(body));
                } else if (starts_with(request, "POST /v1/deployment/activate ")) {
                    response = runtime_http_response(runtime_.activate_deployment(body));
                } else if (starts_with(request, "POST /v1/deployment/drain ")) {
                    response = runtime_http_response(runtime_.drain_deployment(body));
                } else if (starts_with(request, "POST /v1/deployment/unload ")) {
                    response = runtime_http_response(runtime_.unload_deployment(body));
                } else if (starts_with(request, "POST /v1/chat/completions ")) {
                    response = runtime_http_response(runtime_.chat_completion(body));
                } else if (starts_with(request, "POST /v1/layer-split/stage ")) {
                    response = runtime_http_response(runtime_.run_stage(body));
                    keep_alive = requests_stage_keep_alive(request);
                    response.close_connection = !keep_alive;
                }
            }
        } catch (const std::exception& err) {
            response = json_response(
                "400 Bad Request",
                std::string("{\"error\":\"invalid_http_request\",\"message\":\"") + err.what() + "\"}"
            );
        }
        if (!send_response(client_fd, response) || !keep_alive) break;
        set_stage_keep_alive_timeout(client_fd);
    }
}

} // namespace jetsonfabric::runtime
