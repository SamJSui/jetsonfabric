#include "benchmarks/tcp_peer.hpp"

#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::benchmarks {
namespace {

class FileDescriptor {
  public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0) {
            close(value_);
        }
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    FileDescriptor(FileDescriptor&& other) noexcept
        : value_(std::exchange(other.value_, -1)) {}

    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (value_ >= 0) {
                close(value_);
            }
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }

    int get() const { return value_; }
    int release() { return std::exchange(value_, -1); }

  private:
    int value_;
};

void tune_socket(int socket_fd) {
    const int enabled = 1;
    const int buffer_size = 1 << 20;
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0 ||
        setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size)) != 0 ||
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) != 0) {
        throw std::runtime_error("failed to configure TCP socket");
    }
}

using AddressList = std::unique_ptr<addrinfo, decltype(&freeaddrinfo)>;

AddressList resolve_address(
    const std::string& host,
    std::uint16_t port,
    int flags) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = flags;

    addrinfo* result = nullptr;
    const auto service = std::to_string(port);
    const char* node = host.empty() ? nullptr : host.c_str();
    const int status = getaddrinfo(node, service.c_str(), &hints, &result);
    if (status != 0) {
        throw std::runtime_error(std::string("address resolution failed: ") + gai_strerror(status));
    }
    return AddressList(result, freeaddrinfo);
}

}  // namespace

TcpPeer::TcpPeer(int socket_fd) : socket_fd_(socket_fd) {}

TcpPeer::~TcpPeer() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
    }
}

TcpPeer::TcpPeer(TcpPeer&& other) noexcept
    : socket_fd_(std::exchange(other.socket_fd_, -1)) {}

TcpPeer& TcpPeer::operator=(TcpPeer&& other) noexcept {
    if (this != &other) {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
        socket_fd_ = std::exchange(other.socket_fd_, -1);
    }
    return *this;
}

TcpPeer TcpPeer::connect(const std::string& host, std::uint16_t port) {
    const auto addresses = resolve_address(host, port, 0);
    for (auto* address = addresses.get(); address != nullptr; address = address->ai_next) {
        FileDescriptor socket_fd(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (socket_fd.get() < 0) {
            continue;
        }
        if (::connect(socket_fd.get(), address->ai_addr, address->ai_addrlen) == 0) {
            tune_socket(socket_fd.get());
            return TcpPeer(socket_fd.release());
        }
    }
    throw std::runtime_error("failed to connect to peer");
}

TcpPeer TcpPeer::accept(const std::string& listen_address, std::uint16_t port) {
    const auto addresses = resolve_address(listen_address, port, AI_PASSIVE);
    FileDescriptor listener;
    for (auto* address = addresses.get(); address != nullptr; address = address->ai_next) {
        FileDescriptor candidate(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (candidate.get() < 0) {
            continue;
        }
        const int enabled = 1;
        setsockopt(candidate.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (bind(candidate.get(), address->ai_addr, address->ai_addrlen) == 0 &&
            listen(candidate.get(), 1) == 0) {
            listener = std::move(candidate);
            break;
        }
    }
    if (listener.get() < 0) {
        throw std::runtime_error("failed to bind benchmark server");
    }

    FileDescriptor peer(::accept(listener.get(), nullptr, nullptr));
    if (peer.get() < 0) {
        throw std::runtime_error("failed to accept benchmark peer");
    }
    tune_socket(peer.get());
    return TcpPeer(peer.release());
}

void TcpPeer::send_all(const void* data, std::size_t size) const {
    const auto* bytes = static_cast<const std::byte*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const auto result = send(socket_fd_, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (result <= 0) {
            throw std::runtime_error("TCP send failed");
        }
        sent += static_cast<std::size_t>(result);
    }
}

void TcpPeer::receive_all(void* data, std::size_t size) const {
    auto* bytes = static_cast<std::byte*>(data);
    std::size_t received = 0;
    while (received < size) {
        const auto result = recv(socket_fd_, bytes + received, size - received, 0);
        if (result <= 0) {
            throw std::runtime_error("TCP receive failed");
        }
        received += static_cast<std::size_t>(result);
    }
}

}  // namespace jetsonfabric::benchmarks
