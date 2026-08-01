#include "benchmarks/collective_stats.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x4a46434c;
constexpr std::uint32_t kVersion = 1;

enum class Role {
    unset,
    server,
    client,
};

struct Options {
    Role role = Role::unset;
    std::string listen_address = "0.0.0.0";
    std::string peer_address;
    std::uint16_t port = 52500;
    std::size_t elements = 5120;
    std::size_t iterations = 500;
    std::size_t warmup_iterations = 50;
    std::size_t layer_count = 48;
    std::size_t collectives_per_layer = 2;
};

struct WireConfig {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t elements;
    std::uint32_t iterations;
    std::uint32_t warmup_iterations;
};

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

  private:
    int value_;
};

std::size_t parse_size(std::string_view name, const char* value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != std::strlen(value) || parsed == 0 ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive 32-bit integer");
    }
    return static_cast<std::size_t>(parsed);
}

void print_usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --server [options]\n"
        << "  " << program << " --client HOST [options]\n\n"
        << "Options:\n"
        << "  --listen HOST                  server bind address (default 0.0.0.0)\n"
        << "  --port PORT                    TCP port (default 52500)\n"
        << "  --elements COUNT               float elements per rank (default 5120)\n"
        << "  --iterations COUNT             measured collectives (default 500)\n"
        << "  --warmup COUNT                 warmup collectives (default 50)\n"
        << "  --layers COUNT                 projection layer count (default 48)\n"
        << "  --collectives-per-layer COUNT  projection syncs per layer (default 2)\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto require_value = [&]() -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[index];
        };

        if (argument == "--server") {
            options.role = Role::server;
        } else if (argument == "--client") {
            options.role = Role::client;
            options.peer_address = require_value();
        } else if (argument == "--listen") {
            options.listen_address = require_value();
        } else if (argument == "--port") {
            const auto port = parse_size(argument, require_value());
            if (port > std::numeric_limits<std::uint16_t>::max()) {
                throw std::invalid_argument("--port must be between 1 and 65535");
            }
            options.port = static_cast<std::uint16_t>(port);
        } else if (argument == "--elements") {
            options.elements = parse_size(argument, require_value());
        } else if (argument == "--iterations") {
            options.iterations = parse_size(argument, require_value());
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_size(argument, require_value());
        } else if (argument == "--layers") {
            options.layer_count = parse_size(argument, require_value());
        } else if (argument == "--collectives-per-layer") {
            options.collectives_per_layer = parse_size(argument, require_value());
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.role == Role::unset) {
        throw std::invalid_argument("choose --server or --client HOST");
    }
    return options;
}

void tune_socket(int socket_fd) {
    const int enabled = 1;
    const int buffer_size = 1 << 20;
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0 ||
        setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size)) != 0 ||
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) != 0) {
        throw std::runtime_error("failed to configure TCP socket");
    }
}

addrinfo* resolve_address(
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
    return result;
}

FileDescriptor connect_to(const Options& options) {
    addrinfo* raw_addresses = resolve_address(options.peer_address, options.port, 0);
    const std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw_addresses, freeaddrinfo);

    for (auto* address = addresses.get(); address != nullptr; address = address->ai_next) {
        FileDescriptor socket_fd(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (socket_fd.get() < 0) {
            continue;
        }
        if (connect(socket_fd.get(), address->ai_addr, address->ai_addrlen) == 0) {
            tune_socket(socket_fd.get());
            return socket_fd;
        }
    }
    throw std::runtime_error("failed to connect to peer");
}

FileDescriptor accept_peer(const Options& options) {
    addrinfo* raw_addresses = resolve_address(options.listen_address, options.port, AI_PASSIVE);
    const std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw_addresses, freeaddrinfo);

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

    FileDescriptor peer(accept(listener.get(), nullptr, nullptr));
    if (peer.get() < 0) {
        throw std::runtime_error("failed to accept benchmark peer");
    }
    tune_socket(peer.get());
    return peer;
}

void send_all(int socket_fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const auto result = send(socket_fd, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (result <= 0) {
            throw std::runtime_error("TCP send failed");
        }
        sent += static_cast<std::size_t>(result);
    }
}

void receive_all(int socket_fd, void* data, std::size_t size) {
    auto* bytes = static_cast<std::byte*>(data);
    std::size_t received = 0;
    while (received < size) {
        const auto result = recv(socket_fd, bytes + received, size - received, 0);
        if (result <= 0) {
            throw std::runtime_error("TCP receive failed");
        }
        received += static_cast<std::size_t>(result);
    }
}

WireConfig encode_config(const Options& options) {
    return WireConfig{
        .magic = htonl(kMagic),
        .version = htonl(kVersion),
        .elements = htonl(static_cast<std::uint32_t>(options.elements)),
        .iterations = htonl(static_cast<std::uint32_t>(options.iterations)),
        .warmup_iterations = htonl(static_cast<std::uint32_t>(options.warmup_iterations)),
    };
}

Options receive_config(int socket_fd, const Options& server_options) {
    WireConfig wire{};
    receive_all(socket_fd, &wire, sizeof(wire));
    if (ntohl(wire.magic) != kMagic || ntohl(wire.version) != kVersion) {
        throw std::runtime_error("peer sent an unsupported benchmark protocol");
    }

    Options options = server_options;
    options.elements = ntohl(wire.elements);
    options.iterations = ntohl(wire.iterations);
    options.warmup_iterations = ntohl(wire.warmup_iterations);
    if (options.elements == 0 || options.iterations == 0) {
        throw std::runtime_error("peer sent an invalid benchmark configuration");
    }
    return options;
}

void exchange_and_sum(
    int socket_fd,
    const std::vector<float>& local,
    std::vector<float>& remote,
    std::vector<float>& sum) {
    const std::size_t payload_bytes = local.size() * sizeof(float);
    send_all(socket_fd, local.data(), payload_bytes);
    receive_all(socket_fd, remote.data(), payload_bytes);
    for (std::size_t index = 0; index < local.size(); ++index) {
        sum[index] = local[index] + remote[index];
    }
}

void verify_sum(const std::vector<float>& sum) {
    if (sum.empty() || std::abs(sum.front() - 3.0F) > 0.0001F ||
        std::abs(sum.back() - 3.0F) > 0.0001F) {
        throw std::runtime_error("collective sum verification failed");
    }
}

void run_server(const Options& server_options) {
    auto peer = accept_peer(server_options);
    const Options options = receive_config(peer.get(), server_options);
    const std::vector<float> local(options.elements, 1.0F);
    std::vector<float> remote(options.elements);
    std::vector<float> sum(options.elements);

    const std::size_t total_iterations = options.warmup_iterations + options.iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
        exchange_and_sum(peer.get(), local, remote, sum);
    }
    verify_sum(sum);
}

void run_client(const Options& options) {
    auto peer = connect_to(options);
    const WireConfig config = encode_config(options);
    send_all(peer.get(), &config, sizeof(config));

    const std::vector<float> local(options.elements, 2.0F);
    std::vector<float> remote(options.elements);
    std::vector<float> sum(options.elements);
    std::vector<double> latencies_us;
    latencies_us.reserve(options.iterations);

    const std::size_t total_iterations = options.warmup_iterations + options.iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        exchange_and_sum(peer.get(), local, remote, sum);
        const auto stop = std::chrono::steady_clock::now();
        if (iteration >= options.warmup_iterations) {
            latencies_us.push_back(
                std::chrono::duration<double, std::micro>(stop - start).count());
        }
    }
    verify_sum(sum);

    const auto summary = jetsonfabric::benchmarks::summarize_latencies(latencies_us);
    const std::size_t payload_bytes = options.elements * sizeof(float);
    const double aggregate_gbps =
        static_cast<double>(payload_bytes) * 16.0 / (summary.p50_us * 1000.0);
    const double projected_ms = jetsonfabric::benchmarks::projected_communication_ms(
        summary.p50_us,
        options.layer_count,
        options.collectives_per_layer);

    std::cout << std::fixed << std::setprecision(3)
              << "{\n"
              << "  \"operation\": \"two_node_all_reduce_sum\",\n"
              << "  \"transport\": \"persistent_tcp\",\n"
              << "  \"peer\": \"" << options.peer_address << "\",\n"
              << "  \"elements_per_rank\": " << options.elements << ",\n"
              << "  \"payload_bytes_per_rank\": " << payload_bytes << ",\n"
              << "  \"iterations\": " << options.iterations << ",\n"
              << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
              << "  \"mean_us\": " << summary.mean_us << ",\n"
              << "  \"p50_us\": " << summary.p50_us << ",\n"
              << "  \"p95_us\": " << summary.p95_us << ",\n"
              << "  \"p99_us\": " << summary.p99_us << ",\n"
              << "  \"aggregate_effective_gbps_p50\": " << aggregate_gbps << ",\n"
              << "  \"projected_layers\": " << options.layer_count << ",\n"
              << "  \"projected_collectives_per_layer\": "
              << options.collectives_per_layer << ",\n"
              << "  \"projected_communication_ms_per_token_p50\": "
              << projected_ms << "\n"
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.role == Role::server) {
            run_server(options);
        } else {
            run_client(options);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "collective benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
