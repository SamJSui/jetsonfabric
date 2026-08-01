#include "benchmarks/collective_stats.hpp"
#include "benchmarks/tcp_peer.hpp"

#include <arpa/inet.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x4a46544d;
constexpr std::uint32_t kVersion = 1;

enum class Role {
    unset,
    local,
    server,
    client,
};

struct Options {
    Role role = Role::unset;
    std::string listen_address = "0.0.0.0";
    std::string peer_address;
    std::uint16_t port = 52510;
    std::size_t hidden_size = 5120;
    std::size_t intermediate_size = 13824;
    std::size_t iterations = 100;
    std::size_t warmup_iterations = 10;
};

struct WireConfig {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t hidden_size;
    std::uint32_t intermediate_size;
    std::uint32_t iterations;
    std::uint32_t warmup_iterations;
};

struct Timings {
    std::vector<double> compute_us;
    std::vector<double> device_to_host_us;
    std::vector<double> exchange_sum_us;
    std::vector<double> host_to_device_us;
    std::vector<double> all_reduce_us;
    std::vector<double> total_us;
};

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with status " + std::to_string(status));
    }
}

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
        << "  " << program << " --local [options]\n"
        << "  " << program << " --server [options]\n"
        << "  " << program << " --client HOST [options]\n\n"
        << "Options:\n"
        << "  --listen HOST          server bind address (default 0.0.0.0)\n"
        << "  --port PORT            TCP port (default 52510)\n"
        << "  --hidden-size COUNT    model hidden width (default 5120)\n"
        << "  --intermediate COUNT   SwiGLU width (default 13824)\n"
        << "  --iterations COUNT     measured passes (default 100)\n"
        << "  --warmup COUNT         warmup passes (default 10)\n";
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

        if (argument == "--local") {
            options.role = Role::local;
        } else if (argument == "--server") {
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
        } else if (argument == "--hidden-size") {
            options.hidden_size = parse_size(argument, require_value());
        } else if (argument == "--intermediate") {
            options.intermediate_size = parse_size(argument, require_value());
        } else if (argument == "--iterations") {
            options.iterations = parse_size(argument, require_value());
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_size(argument, require_value());
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.role == Role::unset) {
        throw std::invalid_argument("choose --local, --server, or --client HOST");
    }
    if (options.role != Role::local && options.intermediate_size % 2 != 0) {
        throw std::invalid_argument("--intermediate must be divisible by two");
    }
    return options;
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        check_cuda(
            cudaHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(T), cudaHostAllocDefault),
            "cudaHostAlloc");
    }

    ~PinnedBuffer() { cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class CublasHandle {
  public:
    CublasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
    ~CublasHandle() { cublasDestroy(handle_); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

  private:
    cublasHandle_t handle_ = nullptr;
};

__global__ void fill_half(__half* values, std::size_t count, float value) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        values[index] = __float2half(value);
    }
}

__global__ void swiglu(
    const __half* gate,
    const __half* up,
    __half* output,
    std::size_t count) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        const float gate_value = __half2float(gate[index]);
        const float up_value = __half2float(up[index]);
        const float silu = gate_value / (1.0F + expf(-gate_value));
        output[index] = __float2half(silu * up_value);
    }
}

int block_count(std::size_t count) {
    constexpr std::size_t kThreads = 256;
    return static_cast<int>(std::min<std::size_t>((count + kThreads - 1) / kThreads, 65535));
}

class TensorParallelMlpShard {
  public:
    TensorParallelMlpShard(std::size_t hidden_size, std::size_t local_intermediate_size)
        : hidden_size_(hidden_size),
          local_intermediate_size_(local_intermediate_size),
          input_(hidden_size),
          gate_weights_(hidden_size * local_intermediate_size),
          up_weights_(hidden_size * local_intermediate_size),
          down_weights_(hidden_size * local_intermediate_size),
          gate_(local_intermediate_size),
          up_(local_intermediate_size),
          activated_(local_intermediate_size),
          partial_output_(hidden_size),
          reduced_output_(hidden_size) {
        initialize();
    }

    void execute() {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        gemm(
            local_intermediate_size_,
            hidden_size_,
            gate_weights_.data(),
            input_.data(),
            gate_.data(),
            alpha,
            beta);
        gemm(
            local_intermediate_size_,
            hidden_size_,
            up_weights_.data(),
            input_.data(),
            up_.data(),
            alpha,
            beta);

        constexpr int kThreads = 256;
        swiglu<<<block_count(local_intermediate_size_), kThreads>>>(
            gate_.data(),
            up_.data(),
            activated_.data(),
            local_intermediate_size_);
        check_cuda(cudaGetLastError(), "launch SwiGLU kernel");

        gemm(
            hidden_size_,
            local_intermediate_size_,
            down_weights_.data(),
            activated_.data(),
            partial_output_.data(),
            alpha,
            beta);
    }

    __half* partial_output() { return partial_output_.data(); }
    __half* reduced_output() { return reduced_output_.data(); }
    std::size_t output_bytes() const { return hidden_size_ * sizeof(__half); }

  private:
    void initialize() {
        constexpr int kThreads = 256;
        fill_half<<<block_count(input_.size()), kThreads>>>(input_.data(), input_.size(), 0.01F);
        fill_half<<<block_count(gate_weights_.size()), kThreads>>>(
            gate_weights_.data(), gate_weights_.size(), 0.001F);
        fill_half<<<block_count(up_weights_.size()), kThreads>>>(
            up_weights_.data(), up_weights_.size(), 0.001F);
        fill_half<<<block_count(down_weights_.size()), kThreads>>>(
            down_weights_.data(), down_weights_.size(), 0.001F);
        check_cuda(cudaGetLastError(), "initialize MLP buffers");
        check_cuda(cudaDeviceSynchronize(), "synchronize MLP initialization");
    }

    void gemm(
        std::size_t rows,
        std::size_t inner,
        const __half* matrix,
        const __half* vector,
        __half* output,
        float alpha,
        float beta) {
        check_cublas(
            cublasGemmEx(
                handle_.get(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                static_cast<int>(rows),
                1,
                static_cast<int>(inner),
                &alpha,
                matrix,
                CUDA_R_16F,
                static_cast<int>(rows),
                vector,
                CUDA_R_16F,
                static_cast<int>(inner),
                &beta,
                output,
                CUDA_R_16F,
                static_cast<int>(rows),
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx");
    }

    std::size_t hidden_size_;
    std::size_t local_intermediate_size_;
    CublasHandle handle_;
    DeviceBuffer<__half> input_;
    DeviceBuffer<__half> gate_weights_;
    DeviceBuffer<__half> up_weights_;
    DeviceBuffer<__half> down_weights_;
    DeviceBuffer<__half> gate_;
    DeviceBuffer<__half> up_;
    DeviceBuffer<__half> activated_;
    DeviceBuffer<__half> partial_output_;
    DeviceBuffer<__half> reduced_output_;
};

WireConfig encode_config(const Options& options) {
    return WireConfig{
        htonl(kMagic),
        htonl(kVersion),
        htonl(static_cast<std::uint32_t>(options.hidden_size)),
        htonl(static_cast<std::uint32_t>(options.intermediate_size)),
        htonl(static_cast<std::uint32_t>(options.iterations)),
        htonl(static_cast<std::uint32_t>(options.warmup_iterations)),
    };
}

Options receive_config(
    const jetsonfabric::benchmarks::TcpPeer& peer,
    const Options& server_options) {
    WireConfig wire{};
    peer.receive_all(&wire, sizeof(wire));
    if (ntohl(wire.magic) != kMagic || ntohl(wire.version) != kVersion) {
        throw std::runtime_error("peer sent an unsupported MLP benchmark protocol");
    }

    Options options = server_options;
    options.hidden_size = ntohl(wire.hidden_size);
    options.intermediate_size = ntohl(wire.intermediate_size);
    options.iterations = ntohl(wire.iterations);
    options.warmup_iterations = ntohl(wire.warmup_iterations);
    if (options.hidden_size == 0 || options.intermediate_size == 0 ||
        options.iterations == 0 || options.intermediate_size % 2 != 0) {
        throw std::runtime_error("peer sent an invalid MLP benchmark configuration");
    }
    return options;
}

double elapsed_us(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

void exchange_sum(
    const jetsonfabric::benchmarks::TcpPeer& peer,
    PinnedBuffer<__half>& local,
    PinnedBuffer<__half>& remote,
    PinnedBuffer<__half>& sum) {
    const std::size_t payload_bytes = local.size() * sizeof(__half);
    peer.send_all(local.data(), payload_bytes);
    peer.receive_all(remote.data(), payload_bytes);
    for (std::size_t index = 0; index < local.size(); ++index) {
        sum.data()[index] = __float2half(
            __half2float(local.data()[index]) + __half2float(remote.data()[index]));
    }
}

void record(std::vector<double>& samples, double value, bool measured) {
    if (measured) {
        samples.push_back(value);
    }
}

Timings run_benchmark(
    const Options& options,
    std::optional<jetsonfabric::benchmarks::TcpPeer>& peer) {
    const std::size_t world_size = peer.has_value() ? 2 : 1;
    const std::size_t local_intermediate = options.intermediate_size / world_size;
    TensorParallelMlpShard mlp(options.hidden_size, local_intermediate);
    PinnedBuffer<__half> local_output(options.hidden_size);
    PinnedBuffer<__half> remote_output(options.hidden_size);
    PinnedBuffer<__half> reduced_output(options.hidden_size);
    Timings timings;

    const std::size_t total_iterations = options.warmup_iterations + options.iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
        const bool measured = iteration >= options.warmup_iterations;
        const auto total_start = std::chrono::steady_clock::now();

        const auto compute_start = std::chrono::steady_clock::now();
        mlp.execute();
        check_cuda(cudaDeviceSynchronize(), "synchronize MLP execution");
        const auto compute_stop = std::chrono::steady_clock::now();
        record(timings.compute_us, elapsed_us(compute_start, compute_stop), measured);

        if (peer.has_value()) {
            const auto all_reduce_start = std::chrono::steady_clock::now();
            const auto device_to_host_start = all_reduce_start;
            check_cuda(
                cudaMemcpy(
                    local_output.data(),
                    mlp.partial_output(),
                    mlp.output_bytes(),
                    cudaMemcpyDeviceToHost),
                "copy partial output to host");
            const auto device_to_host_stop = std::chrono::steady_clock::now();

            const auto exchange_start = device_to_host_stop;
            exchange_sum(*peer, local_output, remote_output, reduced_output);
            const auto exchange_stop = std::chrono::steady_clock::now();

            const auto host_to_device_start = exchange_stop;
            check_cuda(
                cudaMemcpy(
                    mlp.reduced_output(),
                    reduced_output.data(),
                    mlp.output_bytes(),
                    cudaMemcpyHostToDevice),
                "copy reduced output to device");
            const auto host_to_device_stop = std::chrono::steady_clock::now();

            record(
                timings.device_to_host_us,
                elapsed_us(device_to_host_start, device_to_host_stop),
                measured);
            record(
                timings.exchange_sum_us,
                elapsed_us(exchange_start, exchange_stop),
                measured);
            record(
                timings.host_to_device_us,
                elapsed_us(host_to_device_start, host_to_device_stop),
                measured);
            record(
                timings.all_reduce_us,
                elapsed_us(all_reduce_start, host_to_device_stop),
                measured);
        }

        const auto total_stop = std::chrono::steady_clock::now();
        record(timings.total_us, elapsed_us(total_start, total_stop), measured);
    }

    __half output{};
    if (peer.has_value()) {
        output = reduced_output.data()[0];
        const float expected = 2.0F * __half2float(local_output.data()[0]);
        if (std::abs(__half2float(output) - expected) > 0.01F) {
            throw std::runtime_error("distributed MLP sum verification failed");
        }
    } else {
        check_cuda(
            cudaMemcpy(&output, mlp.partial_output(), sizeof(output), cudaMemcpyDeviceToHost),
            "copy verification output");
    }
    if (!std::isfinite(__half2float(output))) {
        throw std::runtime_error("MLP output verification failed");
    }
    return timings;
}

std::string role_name(Role role) {
    switch (role) {
        case Role::local:
            return "local";
        case Role::server:
            return "server";
        case Role::client:
            return "client";
        case Role::unset:
            break;
    }
    return "unset";
}

void print_summary(const Options& options, const Timings& timings, bool distributed) {
    using jetsonfabric::benchmarks::summarize_latencies;
    const auto compute = summarize_latencies(timings.compute_us);
    const auto total = summarize_latencies(timings.total_us);
    const std::size_t local_intermediate =
        options.intermediate_size / (distributed ? 2 : 1);

    std::cout << std::fixed << std::setprecision(3)
              << "{\n"
              << "  \"operation\": \"qwen_swiglu_mlp_decode\",\n"
              << "  \"role\": \"" << role_name(options.role) << "\",\n"
              << "  \"compute_backend\": \"cuda_fp16_cublas\",\n"
              << "  \"collective_transport\": \""
              << (distributed ? "host_staged_persistent_tcp" : "none") << "\",\n"
              << "  \"hidden_size\": " << options.hidden_size << ",\n"
              << "  \"intermediate_size\": " << options.intermediate_size << ",\n"
              << "  \"local_intermediate_size\": " << local_intermediate << ",\n"
              << "  \"iterations\": " << options.iterations << ",\n"
              << "  \"compute_mean_us\": " << compute.mean_us << ",\n"
              << "  \"compute_p50_us\": " << compute.p50_us << ",\n"
              << "  \"compute_p95_us\": " << compute.p95_us << ",\n";

    if (distributed) {
        const auto device_to_host = summarize_latencies(timings.device_to_host_us);
        const auto exchange = summarize_latencies(timings.exchange_sum_us);
        const auto host_to_device = summarize_latencies(timings.host_to_device_us);
        const auto all_reduce = summarize_latencies(timings.all_reduce_us);
        std::cout
            << "  \"payload_bytes_per_rank\": " << options.hidden_size * sizeof(__half) << ",\n"
            << "  \"device_to_host_p50_us\": " << device_to_host.p50_us << ",\n"
            << "  \"exchange_sum_p50_us\": " << exchange.p50_us << ",\n"
            << "  \"host_to_device_p50_us\": " << host_to_device.p50_us << ",\n"
            << "  \"all_reduce_p50_us\": " << all_reduce.p50_us << ",\n"
            << "  \"all_reduce_p95_us\": " << all_reduce.p95_us << ",\n";
    }

    std::cout << "  \"total_mean_us\": " << total.mean_us << ",\n"
              << "  \"total_p50_us\": " << total.p50_us << ",\n"
              << "  \"total_p95_us\": " << total.p95_us << ",\n"
              << "  \"total_p99_us\": " << total.p99_us << "\n"
              << "}\n";
}

void run(const Options& original_options) {
    Options options = original_options;
    std::optional<jetsonfabric::benchmarks::TcpPeer> peer;
    if (options.role == Role::server) {
        peer.emplace(jetsonfabric::benchmarks::TcpPeer::accept(
            options.listen_address,
            options.port));
        options = receive_config(*peer, options);
    } else if (options.role == Role::client) {
        peer.emplace(jetsonfabric::benchmarks::TcpPeer::connect(
            options.peer_address,
            options.port));
        const WireConfig config = encode_config(options);
        peer->send_all(&config, sizeof(config));
    }

    const Timings timings = run_benchmark(options, peer);
    print_summary(options, timings, peer.has_value());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor-parallel MLP benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
