#include "api/http_response.hpp"

#include <sstream>
#include <utility>

namespace jetsonfabric::runtime {

std::string HttpResponse::serialize() const {
    std::ostringstream out;
    out << "HTTP/1.1 " << status << "\r\n";
    out << "Content-Type: " << content_type << "\r\n";
    out << "Content-Length: " << body.size() << "\r\n";
    out << "Connection: close\r\n";
    out << "\r\n";
    out << body;
    return out.str();
}

HttpResponse json_response(std::string status, std::string body) {
    return HttpResponse{
        .status = std::move(status),
        .content_type = "application/json",
        .body = std::move(body),
    };
}

HttpResponse not_found_response() {
    return json_response("404 Not Found", "{\"error\":\"not found\"}");
}

} // namespace jetsonfabric::runtime
