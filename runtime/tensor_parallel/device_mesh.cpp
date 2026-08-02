#include "tensor_parallel/device_mesh.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace jetsonfabric::runtime::tensor_parallel {
namespace {

std::string trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(first, last - first + 1));
}

std::vector<std::string> split_csv(std::string_view value) {
    std::vector<std::string> parts;
    std::size_t offset = 0;
    while (offset <= value.size()) {
        const std::size_t comma = value.find(',', offset);
        const std::size_t end = comma == std::string_view::npos ? value.size() : comma;
        parts.push_back(trim(value.substr(offset, end - offset)));
        if (comma == std::string_view::npos) {
            break;
        }
        offset = comma + 1;
    }
    return parts;
}

bool valid_port(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    std::uint32_t port = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
    return error == std::errc{} && end == text.data() + text.size() && port > 0 &&
        port <= std::numeric_limits<std::uint16_t>::max();
}

bool valid_endpoint(std::string_view endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        return false;
    }
    const std::string_view host = endpoint.substr(0, colon);
    if (host.find_first_of("/\r\n\t ") != std::string_view::npos) {
        return false;
    }
    return valid_port(endpoint.substr(colon + 1));
}

} // namespace

std::vector<std::string> parse_remote_endpoints(std::string_view value) {
    if (trim(value).empty()) {
        return {};
    }
    return split_csv(value);
}

std::vector<float> parse_tensor_split(std::string_view value) {
    if (trim(value).empty()) {
        return {};
    }
    std::vector<float> split;
    for (const std::string& part : split_csv(value)) {
        if (part.empty()) {
            throw std::invalid_argument("tensor split entries must not be empty");
        }
        std::size_t consumed = 0;
        const float parsed = std::stof(part, &consumed);
        if (consumed != part.size() || !std::isfinite(parsed) || parsed <= 0.0F) {
            throw std::invalid_argument("tensor split entries must be positive finite numbers");
        }
        split.push_back(parsed);
    }
    return split;
}

std::string validate_device_mesh(const DeviceMesh& mesh) {
    if (mesh.transport != kLlamaRpcTransport) {
        return "unsupported tensor transport: " + mesh.transport;
    }
    if (mesh.remote_endpoints.empty()) {
        return "tensor_parallel requires at least one remote RPC endpoint";
    }
    std::set<std::string> unique;
    for (const std::string& endpoint : mesh.remote_endpoints) {
        if (!valid_endpoint(endpoint)) {
            return "invalid tensor RPC endpoint: " + endpoint;
        }
        if (!unique.insert(endpoint).second) {
            return "duplicate tensor RPC endpoint: " + endpoint;
        }
    }
    if (!mesh.tensor_split.empty() && mesh.tensor_split.size() != mesh.world_size()) {
        std::ostringstream message;
        message << "tensor split has " << mesh.tensor_split.size()
                << " entries, want world size " << mesh.world_size();
        return message.str();
    }
    if (std::any_of(mesh.tensor_split.begin(), mesh.tensor_split.end(), [](float value) {
            return !std::isfinite(value) || value <= 0.0F;
        })) {
        return "tensor split entries must be positive finite numbers";
    }
    return {};
}

} // namespace jetsonfabric::runtime::tensor_parallel
