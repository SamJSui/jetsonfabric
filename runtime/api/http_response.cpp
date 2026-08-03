#include "api/http_response.hpp"

#include <sstream>
#include <utility>

namespace jetsonfabric::runtime {

std::size_t HttpResponse::body_size() const noexcept {
    return body.size() + body_payload.size();
}

std::string HttpResponse::serialize_headers() const {
    std::ostringstream out;
    out << "HTTP/1.1 " << status << "\r\n";
    out << "Content-Type: " << content_type << "\r\n";
    out << "Content-Length: " << body_size() << "\r\n";
    out << "Connection: " << (close_connection ? "close" : "keep-alive") << "\r\n";
    out << "\r\n";
    return out.str();
}

std::string HttpResponse::serialize() const {
    std::string wire = serialize_headers();
    wire.reserve(wire.size() + body_size());
    wire.append(body);
    if (!body_payload.empty()) {
        wire.append(
            reinterpret_cast<const char*>(body_payload.data()),
            body_payload.size()
        );
    }
    return wire;
}

HttpResponse json_response(std::string status, std::string body) {
    return HttpResponse{
        std::move(status),
        "application/json",
        std::move(body),
        {},
    };
}

HttpResponse binary_response(
    std::string status,
    std::string content_type,
    std::string body,
    std::vector<std::uint8_t> body_payload
) {
    return HttpResponse{
        std::move(status),
        std::move(content_type),
        std::move(body),
        std::move(body_payload),
    };
}

HttpResponse not_found_response() {
    return json_response("404 Not Found", "{\"error\":\"not found\"}");
}

} // namespace jetsonfabric::runtime
