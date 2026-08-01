#include "benchmarks/collective_stats.hpp"
#include "benchmarks/tcp_peer.hpp"

#include <arpa/inet.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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

WireConfig encode_config(const Options& options) {
    return WireConfig{
        .magic = htonl(kMagic),
        .version = htonl(kVersion),
        .elements = htonl(static_cast<std::uint32_t>(options.elements)),
        .iterations = htonl(static_cast<std::uint32_t>(options.iterations)),
        .warmup_iterations = htonl(static_cast<std::uint32_t>(options.warmup_iterations)),
    };
}

Options receive_config(
    const jetsonfabric::benchmarks::TcpPeer& peer,
    const Options& server_options) {
    WireConfig wire{};
    peer.receive_all(&wire, sizeof(wire));
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
    const jetsonfabric::benchmarks::TcpPeer& peer,
    const std::vector<float>& local,
    std::vector<float>& remote,
    std::vector<float>& sum) {
    const std::size_t payload_bytes = local.size() * sizeof(float);
    peer.send_all(local.data(), payload_bytes);
    peer.receive_all(remote.data(), payload_bytes);
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
    auto peer = jetsonfabric::benchmarks::TcpPeer::accept(
        server_options.listen_address,
        server_options.port);
    const Options options = receive_config(peer, server_options);
    const std::vector<float> local(options.elements, 1.0F);
    std::vector<float> remote(options.elements);
    std::vector<float> sum(options.elements);

    const std::size_t total_iterations = options.warmup_iterations + options.iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
        exchange_and_sum(peer, local, remote, sum);
    }
    verify_sum(sum);
}

void run_client(const Options& options) {
    auto peer = jetsonfabric::benchmarks::TcpPeer::connect(
        options.peer_address,
        options.port);
    const WireConfig config = encode_config(options);
    peer.send_all(&config, sizeof(config));

    const std::vector<float> local(options.elements, 2.0F);
    std::vector<float> remote(options.elements);
    std::vector<float> sum(options.elements);
    std::vector<double> latencies_us;
    latencies_us.reserve(options.iterations);

    const std::size_t total_iterations = options.warmup_iterations + options.iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        exchange_and_sum(peer, local, remote, sum);
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
