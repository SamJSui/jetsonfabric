#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace jetsonfabric::runtime::tensor_parallel {

inline constexpr std::string_view kLlamaRpcTransport = "llama_rpc";

struct DeviceMesh {
    std::string transport = std::string(kLlamaRpcTransport);
    std::vector<std::string> remote_endpoints;
    std::vector<float> tensor_split;

    std::size_t world_size() const noexcept {
        return remote_endpoints.size() + 1;
    }
};

std::vector<std::string> parse_remote_endpoints(std::string_view value);
std::vector<float> parse_tensor_split(std::string_view value);
std::string validate_device_mesh(const DeviceMesh& mesh);

} // namespace jetsonfabric::runtime::tensor_parallel
