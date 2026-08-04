#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jetsonfabric::runtime {

struct HttpResponse {
    std::string status;
    std::string content_type;
    std::string body;
    std::vector<std::uint8_t> body_payload;
    bool close_connection = true;

    std::size_t body_size() const noexcept;
    std::string serialize_headers() const;
    std::string serialize() const;
};

HttpResponse json_response(std::string status, std::string body);
HttpResponse binary_response(
    std::string status,
    std::string content_type,
    std::string body,
    std::vector<std::uint8_t> body_payload = {}
);
HttpResponse not_found_response();

} // namespace jetsonfabric::runtime
