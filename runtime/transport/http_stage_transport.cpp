#include "transport/http_stage_transport.hpp"

#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"
#include "transport/media_type.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

namespace jetsonfabric::runtime::transport {
namespace {

constexpr std::size_t kMaxHTTPResponseBytes = (512U << 20) + (2U << 20);
constexpr auto kConnectionLease = std::chrono::seconds(4);

struct HTTPURL {
    std::string host;
    std::string port = "80";
    std::string path = "/";
};

struct HTTPResponse {
    int status_code = 0;
    std::string status;
    std::map<std::string, std::string> headers;
    std::string body;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

HTTPURL parse_http_url(const std::string& value) {
    constexpr const char* scheme = "http://";
    if (value.rfind(scheme, 0) != 0) {
        throw std::invalid_argument("stage api_url must use http://");
    }
    const std::size_t authority_start = std::strlen(scheme);
    const std::size_t path_start = value.find('/', authority_start);
    const std::string authority = value.substr(
        authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start
    );
    if (authority.empty()) {
        throw std::invalid_argument("stage api_url host is required");
    }

    HTTPURL parsed;
    parsed.path = path_start == std::string::npos ? "/" : value.substr(path_start);
    if (authority.front() == '[') {
        const std::size_t closing = authority.find(']');
        if (closing == std::string::npos) {
            throw std::invalid_argument("invalid bracketed IPv6 stage api_url");
        }
        parsed.host = authority.substr(1, closing - 1);
        if (closing + 1 < authority.size()) {
            if (authority[closing + 1] != ':') {
                throw std::invalid_argument("invalid bracketed IPv6 stage api_url port");
            }
            parsed.port = authority.substr(closing + 2);
        }
    } else {
        const std::size_t colon = authority.rfind(':');
        if (colon != std::string::npos && authority.find(':') == colon) {
            parsed.host = authority.substr(0, colon);
            parsed.port = authority.substr(colon + 1);
        } else {
            parsed.host = authority;
        }
    }
    if (parsed.host.empty() || parsed.port.empty()) {
        throw std::invalid_argument("stage api_url host and port must be non-empty");
    }
    return parsed;
}

std::string stage_path(std::string base) {
    while (base.size() > 1 && base.back() == '/') base.pop_back();
    if (base == "/") base.clear();
    return base + "/v1/layer-split/stage";
}

int connect_socket(const HTTPURL& target) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    const int lookup = getaddrinfo(target.host.c_str(), target.port.c_str(), &hints, &addresses);
    if (lookup != 0) {
        throw std::runtime_error(std::string("resolve stage host: ") + gai_strerror(lookup));
    }

    int connected = -1;
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        int candidate = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (candidate < 0) continue;
        const int flags = fcntl(candidate, F_GETFL, 0);
        if (flags < 0 || fcntl(candidate, F_SETFL, flags | O_NONBLOCK) < 0) {
            close(candidate);
            continue;
        }
        int connect_result = connect(candidate, address->ai_addr, address->ai_addrlen);
        if (connect_result < 0 && errno == EINPROGRESS) {
            fd_set writable;
            FD_ZERO(&writable);
            FD_SET(candidate, &writable);
            timeval connect_timeout{.tv_sec = 5, .tv_usec = 0};
            const int ready = select(candidate + 1, nullptr, &writable, nullptr, &connect_timeout);
            connect_result = -1;
            if (ready > 0) {
                int socket_error = 0;
                socklen_t error_length = sizeof(socket_error);
                if (getsockopt(candidate, SOL_SOCKET, SO_ERROR, &socket_error, &error_length) == 0 &&
                    socket_error == 0) {
                    connect_result = 0;
                }
            }
        }
        if (connect_result == 0 && fcntl(candidate, F_SETFL, flags) == 0) {
            timeval timeout{.tv_sec = 120, .tv_usec = 0};
            setsockopt(candidate, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(candidate, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            connected = candidate;
            break;
        }
        close(candidate);
    }
    freeaddrinfo(addresses);
    if (connected < 0) {
        throw std::runtime_error("connect to stage gateway failed");
    }
    return connected;
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
        if (sent < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("send stage request: ") + std::strerror(errno));
        }
        if (sent == 0) throw std::runtime_error("stage request connection closed while sending");
        offset += static_cast<std::size_t>(sent);
    }
}

std::string decode_chunked(const std::string& encoded) {
    std::string decoded;
    std::size_t cursor = 0;
    while (true) {
        const std::size_t line_end = encoded.find("\r\n", cursor);
        if (line_end == std::string::npos) throw std::runtime_error("truncated chunked stage response");
        std::string size_text = encoded.substr(cursor, line_end - cursor);
        const std::size_t extension = size_text.find(';');
        if (extension != std::string::npos) size_text.resize(extension);
        std::size_t consumed = 0;
        const unsigned long long chunk_size = std::stoull(size_text, &consumed, 16);
        if (consumed != size_text.size()) throw std::runtime_error("invalid stage response chunk size");
        cursor = line_end + 2;
        if (chunk_size == 0) return decoded;
        if (chunk_size > encoded.size() - cursor ||
            cursor + static_cast<std::size_t>(chunk_size) + 2 > encoded.size()) {
            throw std::runtime_error("truncated stage response chunk");
        }
        decoded.append(encoded, cursor, static_cast<std::size_t>(chunk_size));
        cursor += static_cast<std::size_t>(chunk_size);
        if (encoded.compare(cursor, 2, "\r\n") != 0) {
            throw std::runtime_error("stage response chunk omitted its terminator");
        }
        cursor += 2;
    }
}

std::map<std::string, std::string> parse_headers(const std::string& headers) {
    std::map<std::string, std::string> parsed;
    std::istringstream lines(headers);
    std::string line;
    (void) std::getline(lines, line);
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string value = line.substr(colon + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        parsed[lower(line.substr(0, colon))] = std::move(value);
    }
    return parsed;
}

std::optional<std::size_t> chunked_wire_size(const std::string& body) {
    std::size_t cursor = 0;
    while (true) {
        const std::size_t line_end = body.find("\r\n", cursor);
        if (line_end == std::string::npos) return std::nullopt;
        std::string size_text = body.substr(cursor, line_end - cursor);
        const std::size_t extension = size_text.find(';');
        if (extension != std::string::npos) size_text.resize(extension);
        std::size_t consumed = 0;
        const unsigned long long chunk_size = std::stoull(size_text, &consumed, 16);
        if (consumed != size_text.size()) {
            throw std::runtime_error("invalid stage response chunk size");
        }
        if (chunk_size > kMaxHTTPResponseBytes) {
            throw std::runtime_error("stage HTTP response exceeds the configured limit");
        }
        cursor = line_end + 2;
        if (chunk_size == 0) {
            if (body.size() < cursor + 2) return std::nullopt;
            if (body.compare(cursor, 2, "\r\n") == 0) return cursor + 2;
            const std::size_t trailer_end = body.find("\r\n\r\n", cursor);
            return trailer_end == std::string::npos
                ? std::nullopt
                : std::optional<std::size_t>{trailer_end + 4};
        }
        const std::size_t bytes = static_cast<std::size_t>(chunk_size);
        if (body.size() < cursor + bytes + 2) return std::nullopt;
        if (body.compare(cursor + bytes, 2, "\r\n") != 0) {
            throw std::runtime_error("stage response chunk omitted its terminator");
        }
        cursor += bytes + 2;
    }
}

std::string receive_http_response(int socket_fd) {
    std::string response;
    std::size_t header_end = std::string::npos;
    std::optional<std::size_t> content_bytes;
    bool chunked = false;
    char buffer[32 << 10];
    while (true) {
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("receive stage response: ") + std::strerror(errno));
        }
        if (received == 0) {
            throw std::runtime_error("stage response connection closed before a complete response");
        }
        response.append(buffer, static_cast<std::size_t>(received));
        if (response.size() > kMaxHTTPResponseBytes) {
            throw std::runtime_error("stage HTTP response exceeds the configured limit");
        }

        if (header_end == std::string::npos) {
            const std::size_t marker = response.find("\r\n\r\n");
            if (marker == std::string::npos) continue;
            header_end = marker + 4;
            const auto headers = parse_headers(response.substr(0, marker));
            const auto length = headers.find("content-length");
            const auto transfer = headers.find("transfer-encoding");
            if (length != headers.end()) {
                const unsigned long long parsed = std::stoull(length->second);
                if (parsed > kMaxHTTPResponseBytes) {
                    throw std::runtime_error(
                        "stage HTTP response exceeds the configured limit"
                    );
                }
                content_bytes = static_cast<std::size_t>(parsed);
            } else if (
                transfer != headers.end() &&
                lower(transfer->second).find("chunked") != std::string::npos
            ) {
                chunked = true;
            } else {
                throw std::runtime_error(
                    "persistent stage response requires Content-Length or chunked encoding"
                );
            }
        }

        if (content_bytes.has_value() &&
            response.size() >= header_end + *content_bytes) {
            response.resize(header_end + *content_bytes);
            return response;
        }
        if (chunked) {
            const std::optional<std::size_t> body_size =
                chunked_wire_size(response.substr(header_end));
            if (body_size.has_value()) {
                response.resize(header_end + *body_size);
                return response;
            }
        }
    }
}

HTTPResponse parse_http_response(const std::string& wire) {
    const std::size_t header_end = wire.find("\r\n\r\n");
    if (header_end == std::string::npos) throw std::runtime_error("stage response omitted HTTP headers");
    std::istringstream lines(wire.substr(0, header_end));
    std::string line;
    if (!std::getline(lines, line)) throw std::runtime_error("stage response omitted a status line");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t first_space = line.find(' ');
    const std::size_t second_space = line.find(' ', first_space + 1);
    if (first_space == std::string::npos || second_space == std::string::npos) {
        throw std::runtime_error("invalid stage HTTP status line");
    }

    HTTPResponse response;
    response.status_code = std::stoi(line.substr(first_space + 1, second_space - first_space - 1));
    response.status = line.substr(first_space + 1);
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = lower(line.substr(0, colon));
        std::string value = line.substr(colon + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        response.headers[name] = value;
    }
    response.body = wire.substr(header_end + 4);
    const auto transfer = response.headers.find("transfer-encoding");
    if (transfer != response.headers.end() && lower(transfer->second).find("chunked") != std::string::npos) {
        response.body = decode_chunked(response.body);
    } else {
        const auto length = response.headers.find("content-length");
        if (length != response.headers.end() && std::stoull(length->second) != response.body.size()) {
            throw std::runtime_error("stage response Content-Length does not match its body");
        }
    }
    return response;
}

pipeline_parallel::StageRunResult stage_error(
    std::string status,
    std::string code,
    std::string message
) {
    pipeline_parallel::StageRunResult result;
    result.status = std::move(status);
    result.error_code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

} // namespace

class HTTPStageTransport::Impl {
public:
    struct PeerConnection {
        ~PeerConnection() {
            const int fd = socket_fd.exchange(-1);
            if (fd >= 0) close(fd);
        }

        std::mutex mutex;
        std::atomic_int socket_fd{-1};
        std::chrono::steady_clock::time_point last_used;
    };

    Impl(std::string cluster_token, HTTPStageTarget target)
        : cluster_token_(std::move(cluster_token)), target_(target) {}

    std::shared_ptr<PeerConnection> connection_for(const HTTPURL& target) const {
        const auto key = std::make_pair(
            target.host + ':' + target.port,
            std::this_thread::get_id()
        );
        const std::lock_guard lock(connections_mutex_);
        auto [iterator, inserted] = connections_.try_emplace(key);
        if (inserted) {
            iterator->second = std::make_shared<PeerConnection>();
        }
        return iterator->second;
    }

    const std::string& cluster_token() const noexcept {
        return cluster_token_;
    }

    const std::string& target_url(const protocol::GenerationStage& stage) const {
        if (target_ == HTTPStageTarget::Runtime) {
            if (stage.runtime_url.empty()) {
                throw std::invalid_argument("direct stage transport requires runtime_url");
            }
            return stage.runtime_url;
        }
        return stage.api_url;
    }

    bool uses_runtime_target() const noexcept {
        return target_ == HTTPStageTarget::Runtime;
    }

    void reset(const std::shared_ptr<PeerConnection>& connection) const noexcept {
        const int fd = connection->socket_fd.exchange(-1);
        if (fd >= 0) close(fd);
        connection->last_used = {};
    }

    void interrupt_all() const noexcept {
        const std::lock_guard lock(connections_mutex_);
        for (const auto& [_, connection] : connections_) {
            const int fd = connection->socket_fd.load();
            if (fd >= 0) {
                (void) ::shutdown(fd, SHUT_RDWR);
            }
        }
    }

private:
    using ConnectionKey = std::pair<std::string, std::thread::id>;

    std::string cluster_token_;
    HTTPStageTarget target_;
    mutable std::mutex connections_mutex_;
    mutable std::map<ConnectionKey, std::shared_ptr<PeerConnection>> connections_;
};

HTTPStageTransport::HTTPStageTransport(
    std::string cluster_token,
    HTTPStageTarget target
)
    : impl_(std::make_unique<Impl>(std::move(cluster_token), target)) {}

HTTPStageTransport::~HTTPStageTransport() = default;

void HTTPStageTransport::shutdown() const {
    impl_->interrupt_all();
}

pipeline_parallel::StageRunResult HTTPStageTransport::invoke(
    const protocol::GenerationStage& stage,
    const protocol::StageRequest& request,
    pipeline_parallel::StageOperation operation
) const {
    if (impl_->cluster_token().empty()) {
        return stage_error(
            "503 Service Unavailable",
            "cluster_auth_unconfigured",
            "runtime peer transport requires JETSONFABRIC_CLUSTER_TOKEN"
        );
    }
    try {
        const auto call_start = std::chrono::steady_clock::now();
        const HTTPURL target = parse_http_url(impl_->target_url(stage));
        if (impl_->uses_runtime_target()) {
            const HTTPURL node_api = parse_http_url(stage.api_url);
            if (lower(target.host) != lower(node_api.host)) {
                return stage_error(
                    "400 Bad Request",
                    "runtime_endpoint_mismatch",
                    "direct runtime host must match the stage node API host"
                );
            }
        }
        const std::string operation_name = operation == pipeline_parallel::StageOperation::CloseSession
            ? protocol::kStageOperationCloseSession
            : operation == pipeline_parallel::StageOperation::RollbackSession
                ? protocol::kStageOperationRollbackSession
                : protocol::kStageOperationExecute;
        const std::string body = protocol::encode_stage_request(request, operation_name);
        std::ostringstream headers;
        headers << "POST " << stage_path(target.path) << " HTTP/1.1\r\n"
                << "Host: " << target.host << ':' << target.port << "\r\n"
                << "Content-Type: " << protocol::kStageWireContentType << "\r\n"
                << "Accept: " << protocol::kStageWireContentType << "\r\n"
                << "X-JetsonFabric-Cluster-Token: " << impl_->cluster_token() << "\r\n"
                << "Content-Length: " << body.size() << "\r\n"
                << "Connection: keep-alive\r\n\r\n";

        const std::shared_ptr<Impl::PeerConnection> connection =
            impl_->connection_for(target);
        const std::lock_guard connection_lock(connection->mutex);
        try {
            const auto now = std::chrono::steady_clock::now();
            if (connection->socket_fd.load() >= 0 &&
                now - connection->last_used >= kConnectionLease) {
                impl_->reset(connection);
            }
            if (connection->socket_fd.load() < 0) {
                connection->socket_fd.store(connect_socket(target));
            }
            const int fd = connection->socket_fd.load();
            send_all(fd, headers.str());
            send_all(fd, body);
        } catch (...) {
            impl_->reset(connection);
            throw;
        }
        std::string wire_response;
        try {
            wire_response = receive_http_response(connection->socket_fd.load());
            connection->last_used = std::chrono::steady_clock::now();
        } catch (...) {
            impl_->reset(connection);
            throw;
        }
        HTTPResponse response;
        try {
            response = parse_http_response(wire_response);
        } catch (...) {
            impl_->reset(connection);
            throw;
        }
        const auto connection_header = response.headers.find("connection");
        if (connection_header != response.headers.end() &&
            lower(connection_header->second).find("close") != std::string::npos) {
            impl_->reset(connection);
        }
        const auto content_type = response.headers.find("content-type");
        if (content_type == response.headers.end() ||
            !matches_media_type(content_type->second, protocol::kStageWireContentType)) {
            std::string code = "runtime_stage_failed";
            std::string message = response.body;
            try {
                const nlohmann::json error = nlohmann::json::parse(response.body);
                code = error.value("error", code);
                message = error.value("message", message);
            } catch (const nlohmann::json::exception&) {
            }
            return stage_error(response.status, code, message);
        }
        protocol::StageResponse decoded = protocol::decode_stage_response(response.body);
        if (decoded.operation != operation_name) {
            return stage_error(
                "502 Bad Gateway",
                "runtime_stage_operation_mismatch",
                "peer runtime acknowledged a different stage operation"
            );
        }
        if (response.status_code < 200 || response.status_code >= 300 || !decoded.error.empty()) {
            return stage_error(
                response.status,
                decoded.error.empty() ? "runtime_stage_failed" : decoded.error,
                decoded.message
            );
        }
        pipeline_parallel::StageRunResult result;
        result.ok = true;
        result.status = response.status;
        result.response = std::move(decoded);
        result.remote_call_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - call_start
        ).count();
        return result;
    } catch (const std::exception& error) {
        return stage_error("502 Bad Gateway", "runtime_stage_transport_failed", error.what());
    }
}

} // namespace jetsonfabric::runtime::transport
