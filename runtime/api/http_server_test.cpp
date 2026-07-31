#include "api/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

std::uint16_t reserve_port() {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) throw std::runtime_error("test socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        close(socket_fd);
        throw std::runtime_error("test port bind failed");
    }
    socklen_t length = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &length) < 0) {
        close(socket_fd);
        throw std::runtime_error("test port lookup failed");
    }
    close(socket_fd);
    return ntohs(address.sin_port);
}

int connect_with_retry(std::uint16_t port) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) throw std::runtime_error("test client socket failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            timeval timeout{.tv_sec = 2, .tv_usec = 0};
            setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            return socket_fd;
        }
        close(socket_fd);
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error("test client could not connect to runtime server");
}

void send_wire(int socket_fd, const std::string& wire_request) {
    std::size_t offset = 0;
    while (offset < wire_request.size()) {
        const ssize_t sent = send(
            socket_fd,
            wire_request.data() + offset,
            wire_request.size() - offset,
            MSG_NOSIGNAL
        );
        if (sent <= 0) throw std::runtime_error("test client send failed");
        offset += static_cast<std::size_t>(sent);
    }
}

std::size_t response_content_length(const std::string& headers) {
    constexpr const char* name = "Content-Length:";
    const std::size_t start = headers.find(name);
    if (start == std::string::npos) throw std::runtime_error("response omitted Content-Length");
    const std::size_t value_start = start + std::strlen(name);
    const std::size_t value_end = headers.find("\r\n", value_start);
    return std::stoull(headers.substr(value_start, value_end - value_start));
}

std::string read_response(int socket_fd) {
    std::string response;
    std::size_t expected_size = 0;
    char buffer[4096];
    while (expected_size == 0 || response.size() < expected_size) {
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) throw std::runtime_error("test client recv failed");
        response.append(buffer, static_cast<std::size_t>(received));
        const std::size_t marker = response.find("\r\n\r\n");
        if (marker != std::string::npos && expected_size == 0) {
            expected_size = marker + 4 + response_content_length(response.substr(0, marker));
        }
    }
    return response.substr(0, expected_size);
}

std::string read_responses(int socket_fd, std::size_t response_count) {
    std::string responses;
    char buffer[4096];
    while (true) {
        std::size_t found = 0;
        std::size_t cursor = 0;
        while ((cursor = responses.find("HTTP/1.1 ", cursor)) != std::string::npos) {
            ++found;
            cursor += 9;
        }
        if (found >= response_count) return responses;

        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) throw std::runtime_error("test client recv failed");
        responses.append(buffer, static_cast<std::size_t>(received));
    }
}

std::string request(std::uint16_t port, const std::string& wire_request) {
    const int socket_fd = connect_with_retry(port);
    send_wire(socket_fd, wire_request);

    std::string response;
    char buffer[4096];
    while (true) {
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received < 0) {
            close(socket_fd);
            throw std::runtime_error(
                std::string("test client recv failed: ") + std::strerror(errno)
            );
        }
        if (received == 0) break;
        response.append(buffer, static_cast<std::size_t>(received));
    }
    close(socket_fd);
    return response;
}

class BlockingRuntime final : public jetsonfabric::runtime::RuntimeAPI {
public:
    std::string runtime_name() const override { return "test-runtime"; }
    std::string engine_name() const override { return "test-engine"; }
    jetsonfabric::runtime::ExecutionMode execution_mode() const override {
        return jetsonfabric::runtime::ExecutionMode::PipelineParallel;
    }
    std::string model() const override { return "test-model"; }
    std::string stage_transport_name() const override { return "test-transport"; }
    std::string activation_encoding() const override { return "test-encoding"; }
    std::string kv_cache_type() const override { return "test-kv-cache"; }
    int ubatch_size() const override { return 128; }

    jetsonfabric::runtime::RuntimeResponse deployment_status() const override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse load_deployment(const std::string&) override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse activate_deployment(const std::string&) override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse drain_deployment(const std::string&) override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse unload_deployment(const std::string&) override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse chat_completion(const std::string&) const override {
        return ok();
    }
    jetsonfabric::runtime::RuntimeResponse run_stage(const std::string&) const override {
        return ok();
    }

    jetsonfabric::runtime::RuntimeResponse generate(
        const std::string&,
        const jetsonfabric::runtime::GenerationEventSink&
    ) const override {
        std::unique_lock lock(mutex_);
        generation_started_ = true;
        changed_.notify_all();
        changed_.wait(lock, [this]() { return release_generation_; });
        return {
            "200 OK",
            "application/x-ndjson",
            "{\"event\":\"done\"}",
        };
    }

    void wait_for_generation() const {
        std::unique_lock lock(mutex_);
        if (!changed_.wait_for(lock, 2s, [this]() { return generation_started_; })) {
            throw std::runtime_error("generation did not reach the runtime");
        }
    }

    void release_generation() const {
        const std::lock_guard lock(mutex_);
        release_generation_ = true;
        changed_.notify_all();
    }

private:
    static jetsonfabric::runtime::RuntimeResponse ok() {
        return {"200 OK", "application/json", "{}"};
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    mutable bool generation_started_ = false;
    mutable bool release_generation_ = false;
};

} // namespace

int main() {
    using namespace jetsonfabric::runtime;

    const std::uint16_t port = reserve_port();
    Config config;
    config.host = "127.0.0.1";
    config.port = port;
    config.http_workers = 2;
    std::atomic_bool running{true};
    BlockingRuntime runtime;
    HttpServer server(config, runtime, running);
    std::thread server_thread([&]() { (void) server.run(); });

    std::string generation_response;
    std::thread generation_client([&]() {
        generation_response = request(
            port,
            "POST /v1/generate HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Length: 2\r\n"
            "Connection: close\r\n\r\n{}"
        );
    });
    runtime.wait_for_generation();

    std::future<std::string> health = std::async(std::launch::async, [port]() {
        return request(
            port,
            "GET /healthz HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Connection: close\r\n\r\n"
        );
    });
    const bool health_ready = health.wait_for(1s) == std::future_status::ready;

    runtime.release_generation();
    generation_client.join();
    const std::string health_response = health.get();

    const int persistent_fd = connect_with_retry(port);
    const std::string stage_request =
        "POST /v1/layer-split/stage HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Length: 2\r\n"
        "Connection: keep-alive\r\n\r\n{}";
    send_wire(persistent_fd, stage_request);
    const std::string first_stage_response = read_response(persistent_fd);
    send_wire(persistent_fd, stage_request);
    const std::string second_stage_response = read_response(persistent_fd);
    close(persistent_fd);

    const int pipelined_fd = connect_with_retry(port);
    send_wire(pipelined_fd, stage_request + stage_request);
    const std::string pipelined_responses = read_responses(pipelined_fd, 2);
    close(pipelined_fd);

    const int idle_fd = connect_with_retry(port);
    std::this_thread::sleep_for(50ms);
    const auto shutdown_started = std::chrono::steady_clock::now();
    running.store(false);
    server_thread.join();
    const auto shutdown_elapsed =
        std::chrono::steady_clock::now() - shutdown_started;
    close(idle_fd);

    expect(health_ready, "health request waited behind an in-flight generation");
    expect(
        health_response.find("200 OK") != std::string::npos,
        "concurrent health request failed"
    );
    expect(
        health_response.find("\"stage_transport\":\"test-transport\"") !=
                std::string::npos &&
            health_response.find("\"activation_encoding\":\"test-encoding\"") !=
                std::string::npos &&
            health_response.find("\"kv_cache_type\":\"test-kv-cache\"") !=
                std::string::npos &&
            health_response.find("\"ubatch_size\":128") != std::string::npos,
        "runtime health omitted transport compatibility"
    );
    expect(
        generation_response.find("\"event\":\"done\"") != std::string::npos,
        "generation request did not complete"
    );
    expect(
        first_stage_response.find("Connection: keep-alive") != std::string::npos &&
            second_stage_response.find("Connection: keep-alive") != std::string::npos,
        "stage requests did not reuse a persistent connection"
    );
    expect(
        pipelined_responses.find("HTTP/1.1 200 OK") != std::string::npos &&
            pipelined_responses.rfind("HTTP/1.1 200 OK") !=
                pipelined_responses.find("HTTP/1.1 200 OK"),
        "pipelined stage requests did not both receive responses"
    );
    expect(
        shutdown_elapsed < 1s,
        "idle client prevented prompt HTTP server shutdown"
    );
    std::cout << "HTTP server concurrency tests passed\n";
    return 0;
}
