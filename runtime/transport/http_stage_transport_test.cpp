#include "transport/http_stage_transport.hpp"

#include "protocol/stage.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

void send_all(int socket_fd, const std::string& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t sent = send(
            socket_fd,
            data.data() + offset,
            data.size() - offset,
            MSG_NOSIGNAL
        );
        if (sent <= 0) {
            throw std::runtime_error(
                std::string("test server send failed: ") + std::strerror(errno)
            );
        }
        offset += static_cast<std::size_t>(sent);
    }
}

std::size_t content_length(const std::string& headers) {
    constexpr const char* name = "Content-Length:";
    const std::size_t start = headers.find(name);
    if (start == std::string::npos) {
        throw std::runtime_error("test request omitted Content-Length");
    }
    const std::size_t value_start = start + std::strlen(name);
    const std::size_t value_end = headers.find("\r\n", value_start);
    return std::stoull(headers.substr(value_start, value_end - value_start));
}

std::string read_request(int socket_fd) {
    std::string request;
    char buffer[4096];
    std::size_t header_end = std::string::npos;
    std::size_t body_size = 0;
    while (true) {
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("test server recv failed: ") + std::strerror(errno)
            );
        }
        if (received == 0) return {};
        request.append(buffer, static_cast<std::size_t>(received));
        if (header_end == std::string::npos) {
            const std::size_t marker = request.find("\r\n\r\n");
            if (marker == std::string::npos) continue;
            header_end = marker + 4;
            body_size = content_length(request.substr(0, marker));
        }
        if (request.size() >= header_end + body_size) {
            return request.substr(0, header_end + body_size);
        }
    }
}

jetsonfabric::runtime::protocol::StageResponse response_for(
    const jetsonfabric::runtime::protocol::StageRequest& request
) {
    jetsonfabric::runtime::protocol::StageResponse response;
    response.session_id = request.session_id;
    response.request_id = request.request_id;
    response.model_id = request.model_id;
    response.phase = request.phase;
    response.decode_step = request.decode_step;
    response.stage_index = request.stage_index;
    response.stage_count = request.stage_count;
    response.node_name = request.node_name;
    response.layer_start = request.layer_start;
    response.layer_end = request.layer_end;
    response.payload_kind = "text";
    response.encoding = "utf-8";
    response.message = "ok";
    return response;
}

int open_server(std::uint16_t& port) {
    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) throw std::runtime_error("test server socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(server_fd, 2) < 0) {
        close(server_fd);
        throw std::runtime_error("test server bind or listen failed");
    }
    socklen_t length = sizeof(address);
    if (getsockname(server_fd, reinterpret_cast<sockaddr*>(&address), &length) < 0) {
        close(server_fd);
        throw std::runtime_error("test server getsockname failed");
    }
    port = ntohs(address.sin_port);
    return server_fd;
}

} // namespace

int main() {
    using namespace jetsonfabric::runtime;

    std::uint16_t port = 0;
    const int server_fd = open_server(port);
    int accepted_connections = 0;
    std::string server_error;
    std::thread server([&]() {
        int client_fd = -1;
        try {
            for (int index = 0; index < 2; ++index) {
                std::string request;
                while (request.empty()) {
                    if (client_fd < 0) {
                        client_fd = accept(server_fd, nullptr, nullptr);
                        if (client_fd < 0) {
                            throw std::runtime_error("test server accept failed");
                        }
                        ++accepted_connections;
                    }
                    request = read_request(client_fd);
                    if (request.empty()) {
                        close(client_fd);
                        client_fd = -1;
                    }
                }

                const std::size_t marker = request.find("\r\n\r\n");
                const protocol::StageRequest decoded =
                    protocol::decode_stage_request(request.substr(marker + 4));
                const std::string body =
                    protocol::encode_stage_response(response_for(decoded));
                std::ostringstream response;
                response << "HTTP/1.1 200 OK\r\n"
                         << "Content-Type: " << protocol::kStageWireContentType << "\r\n"
                         << "Content-Length: " << body.size() << "\r\n"
                         << "Connection: " << (index == 1 ? "close" : "keep-alive")
                         << "\r\n\r\n"
                         << body;
                send_all(client_fd, response.str());
            }
        } catch (const std::exception& error) {
            server_error = error.what();
        }
        if (client_fd >= 0) close(client_fd);
        close(server_fd);
    });

    transport::HTTPStageTransport stage_transport("test-token");
    protocol::GenerationStage stage{
        .stage_index = 1,
        .stage_count = 2,
        .node_id = "node-b",
        .node_name = "node-b",
        .api_url = "http://127.0.0.1:" + std::to_string(port),
        .layer_start = 4,
        .layer_end = 8,
    };
    protocol::StageRequest request;
    request.session_id = "session-1";
    request.request_id = "request-1";
    request.model_id = "model-a";
    request.stage_index = 1;
    request.stage_count = 2;
    request.node_name = "node-b";
    request.layer_start = 4;
    request.layer_end = 8;
    request.payload = {'h', 'i'};

    const pipeline_parallel::StageRunResult first = stage_transport.invoke(
        stage,
        request,
        pipeline_parallel::StageOperation::Execute
    );
    request.request_id = "request-2";
    const pipeline_parallel::StageRunResult second = stage_transport.invoke(
        stage,
        request,
        pipeline_parallel::StageOperation::Execute
    );
    server.join();

    expect(server_error.empty(), "test server failed: " + server_error);
    expect(first.ok && second.ok, "persistent stage requests did not both succeed");
    expect(accepted_connections == 1, "stage transport did not reuse its peer connection");
    std::cout << "HTTP stage transport tests passed\n";
    return 0;
}
