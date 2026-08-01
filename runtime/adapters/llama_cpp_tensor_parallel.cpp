#include "adapters/llama_cpp_tensor_parallel.hpp"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

using AddRpcServer = ggml_backend_reg_t (*)(const char* endpoint);

std::mutex rpc_registry_mutex;
std::map<std::string, std::vector<ggml_backend_dev_t>> rpc_devices_by_endpoint;

bool is_cuda_device(ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }
    const std::string registration = ggml_backend_reg_name(
        ggml_backend_dev_backend_reg(device)
    );
    return registration.starts_with("CUDA");
}

ggml_backend_dev_t local_cuda_device() {
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (is_cuda_device(device)) {
            return device;
        }
    }
    throw std::runtime_error("tensor_parallel requires a local CUDA device");
}

std::vector<ggml_backend_dev_t> register_rpc_endpoint(const std::string& endpoint) {
    std::lock_guard lock(rpc_registry_mutex);
    const auto existing = rpc_devices_by_endpoint.find(endpoint);
    if (existing != rpc_devices_by_endpoint.end()) {
        return existing->second;
    }

    ggml_backend_reg_t rpc = ggml_backend_reg_by_name("RPC");
    if (rpc == nullptr) {
        throw std::runtime_error("llama.cpp RPC backend is not available in this build");
    }
    auto add_server = reinterpret_cast<AddRpcServer>(
        ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server")
    );
    if (add_server == nullptr) {
        throw std::runtime_error("llama.cpp RPC backend does not expose device registration");
    }

    ggml_backend_reg_t remote = add_server(endpoint.c_str());
    if (remote == nullptr) {
        throw std::runtime_error("failed to register tensor RPC endpoint " + endpoint);
    }
    ggml_backend_register(remote);

    std::vector<ggml_backend_dev_t> devices;
    for (std::size_t index = 0; index < ggml_backend_reg_dev_count(remote); ++index) {
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(remote, index);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(device);
        }
    }
    if (devices.empty()) {
        throw std::runtime_error("tensor RPC endpoint exposes no accelerator: " + endpoint);
    }
    rpc_devices_by_endpoint.emplace(endpoint, devices);
    return devices;
}

} // namespace

class LlamaCppTensorParallel::Impl {
public:
    explicit Impl(tensor_parallel::DeviceMesh mesh_in)
        : mesh(std::move(mesh_in)) {
        const std::string error = tensor_parallel::validate_device_mesh(mesh);
        if (!error.empty()) {
            throw std::invalid_argument(error);
        }

        devices.push_back(local_cuda_device());
        for (const std::string& endpoint : mesh.remote_endpoints) {
            std::vector<ggml_backend_dev_t> remote = register_rpc_endpoint(endpoint);
            devices.push_back(remote.front());
        }
        if (devices.size() > llama_max_devices()) {
            throw std::runtime_error("tensor device mesh exceeds llama.cpp device capacity");
        }
        if (!mesh.tensor_split.empty()) {
            tensor_split.assign(llama_max_devices(), 0.0F);
            std::copy(mesh.tensor_split.begin(), mesh.tensor_split.end(), tensor_split.begin());
        }
        devices.push_back(nullptr);
    }

    tensor_parallel::DeviceMesh mesh;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> tensor_split;
};

LlamaCppTensorParallel::LlamaCppTensorParallel(tensor_parallel::DeviceMesh mesh)
    : impl_(std::make_unique<Impl>(std::move(mesh))) {}

LlamaCppTensorParallel::~LlamaCppTensorParallel() = default;

void LlamaCppTensorParallel::configure(llama_model_params& params) const {
    params.devices = impl_->devices.data();
    params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    params.tensor_split = impl_->tensor_split.empty()
        ? nullptr
        : impl_->tensor_split.data();
}

std::size_t LlamaCppTensorParallel::world_size() const noexcept {
    return impl_->devices.size() - 1;
}

} // namespace jetsonfabric::runtime::adapters
