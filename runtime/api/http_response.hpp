#pragma once

#include <string>

namespace jetsonfabric::runtime {

struct HttpResponse {
    std::string status;
    std::string content_type;
    std::string body;

    std::string serialize() const;
};

HttpResponse json_response(std::string status, std::string body);
HttpResponse binary_response(std::string status, std::string content_type, std::string body);
HttpResponse not_found_response();

} // namespace jetsonfabric::runtime
