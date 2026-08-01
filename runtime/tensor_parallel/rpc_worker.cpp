#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using StartRpcServer = void (*)(
    const char*,
    const char*,
    std::size_t,
    std::size_t,
    ggml_backend_dev_t*
);

struct Options {
    std::string endpoint = "127.0.0.1:52520";
    std::string cache_dir;
    int threads = 0;
    bool allow_remote = false;
};

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Exposes this host's CUDA device to a trusted JetsonFabric tensor driver.\n"
        << "GGML RPC has no transport authentication; never expose it to an untrusted network.\n\n"
        << "  --listen host:port   RPC endpoint, default 127.0.0.1:52520\n"
        << "  --cache-dir path     optional persistent tensor cache\n"
        << "  --threads count      RPC worker threads, default hardware concurrency\n"
        << "  --allow-remote       acknowledge unauthenticated non-loopback exposure\n";
}

int parse_positive_int(std::string_view value, std::string_view flag) {
    std::size_t consumed = 0;
    const int parsed = std::stoi(std::string(value), &consumed);
    if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(std::string(flag) + " must be a positive integer");
    }
    return parsed;
}

std::string endpoint_host(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        throw std::invalid_argument("--listen must be host:port");
    }
    return std::string(endpoint.substr(0, colon));
}

bool is_loopback(std::string_view host) {
    return host == "localhost" || host == "127.0.0.1" || host == "[::1]";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto value = [&]() -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--listen") {
            options.endpoint = value();
        } else if (argument == "--cache-dir") {
            options.cache_dir = value();
        } else if (argument == "--threads") {
            options.threads = parse_positive_int(value(), argument);
        } else if (argument == "--allow-remote") {
            options.allow_remote = true;
        } else if (argument == "--help" || argument == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    const std::string host = endpoint_host(options.endpoint);
    if (!is_loopback(host) && !options.allow_remote) {
        throw std::invalid_argument(
            "non-loopback --listen requires --allow-remote because GGML RPC is unauthenticated"
        );
    }
    if (options.threads == 0) {
        options.threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    }
    return options;
}

std::vector<ggml_backend_dev_t> cuda_devices() {
    std::vector<ggml_backend_dev_t> devices;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        const std::string registration = ggml_backend_reg_name(
            ggml_backend_dev_backend_reg(device)
        );
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU ||
             type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            registration.starts_with("CUDA")) {
            devices.push_back(device);
        }
    }
    return devices;
}

StartRpcServer rpc_server_entrypoint() {
    ggml_backend_reg_t rpc = ggml_backend_reg_by_name("RPC");
    if (rpc == nullptr) {
        throw std::runtime_error("GGML RPC backend is not available in this build");
    }
    auto start_server = reinterpret_cast<StartRpcServer>(
        ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_start_server")
    );
    if (start_server == nullptr) {
        throw std::runtime_error("GGML RPC backend does not expose its server entrypoint");
    }
    return start_server;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        ggml_backend_load_all();
        std::vector<ggml_backend_dev_t> devices = cuda_devices();
        if (devices.empty()) {
            throw std::runtime_error("no CUDA accelerator is available to the tensor worker");
        }
        std::cerr
            << "JetsonFabric tensor worker listening on " << options.endpoint
            << " devices=" << devices.size()
            << " transport=llama_rpc trusted_lan_only=true\n";
        rpc_server_entrypoint()(
            options.endpoint.c_str(),
            options.cache_dir.empty() ? nullptr : options.cache_dir.c_str(),
            options.threads,
            devices.size(),
            devices.data()
        );
        throw std::runtime_error("GGML RPC server stopped unexpectedly");
    } catch (const std::exception& error) {
        std::cerr << "jetsonfabric-tensor-worker: " << error.what() << "\n";
        return 2;
    }
}
