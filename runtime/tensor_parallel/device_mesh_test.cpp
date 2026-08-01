#include "tensor_parallel/device_mesh.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tensor_parallel = jetsonfabric::runtime::tensor_parallel;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        const auto endpoints = tensor_parallel::parse_remote_endpoints(
            " 192.168.1.65:52520,grumpy.local:52521 "
        );
        require(endpoints.size() == 2, "expected two parsed endpoints");
        require(endpoints[0] == "192.168.1.65:52520", "first endpoint was not trimmed");

        const auto split = tensor_parallel::parse_tensor_split("1, 1.5,2");
        require(split.size() == 3, "expected three tensor split entries");
        require(std::abs(split[1] - 1.5F) < 0.001F, "tensor split value changed");

        tensor_parallel::DeviceMesh mesh{
            .remote_endpoints = {"192.168.1.65:52520"},
            .tensor_split = {1.0F, 1.0F},
        };
        require(tensor_parallel::validate_device_mesh(mesh).empty(), "valid mesh was rejected");
        require(mesh.world_size() == 2, "world size must include the local CUDA device");

        mesh.remote_endpoints.push_back("192.168.1.65:52520");
        require(
            tensor_parallel::validate_device_mesh(mesh).find("duplicate") != std::string::npos,
            "duplicate endpoint was accepted"
        );
        std::cout << "tensor device mesh tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tensor device mesh tests failed: " << error.what() << "\n";
        return 1;
    }
}
