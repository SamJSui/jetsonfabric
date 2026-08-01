#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace jetsonfabric::benchmarks {

class TcpPeer {
  public:
    static TcpPeer connect(const std::string& host, std::uint16_t port);
    static TcpPeer accept(const std::string& listen_address, std::uint16_t port);

    ~TcpPeer();
    TcpPeer(const TcpPeer&) = delete;
    TcpPeer& operator=(const TcpPeer&) = delete;
    TcpPeer(TcpPeer&& other) noexcept;
    TcpPeer& operator=(TcpPeer&& other) noexcept;

    void send_all(const void* data, std::size_t size) const;
    void receive_all(void* data, std::size_t size) const;

  private:
    explicit TcpPeer(int socket_fd);

    int socket_fd_ = -1;
};

}  // namespace jetsonfabric::benchmarks
