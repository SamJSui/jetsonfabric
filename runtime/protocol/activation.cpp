#include "protocol/activation.hpp"

#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::protocol {
namespace {

std::size_t skip_ws(const std::string& text, std::size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

void require_json_object(const std::string& json) {
    const std::size_t start = skip_ws(json, 0);
    if (start >= json.size() || json[start] != '{') {
        throw std::invalid_argument("activation request must be a JSON object");
    }
}

std::size_t field_value_pos(const std::string& json, const std::string& field) {
    const std::string key = "\"" + field + "\"";
    const std::size_t key_pos = json.find(key);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }

    const std::size_t colon = json.find(':', key_pos + key.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }

    return skip_ws(json, colon + 1);
}

std::string read_json_string_at(const std::string& json, std::size_t pos, const std::string& fallback) {
    if (pos == std::string::npos || pos >= json.size() || json[pos] != '"') {
        return fallback;
    }

    std::string output;

    for (std::size_t i = pos + 1; i < json.size(); ++i) {
        const char c = json[i];

        if (c == '\\' && i + 1 < json.size()) {
            const char escaped = json[++i];

            switch (escaped) {
            case '"':
                output.push_back('"');
                break;
            case '\\':
                output.push_back('\\');
                break;
            case '/':
                output.push_back('/');
                break;
            case 'n':
                output.push_back('\n');
                break;
            case 'r':
                output.push_back('\r');
                break;
            case 't':
                output.push_back('\t');
                break;
            default:
                output.push_back(escaped);
                break;
            }

            continue;
        }

        if (c == '"') {
            return output;
        }

        output.push_back(c);
    }

    throw std::invalid_argument("unterminated JSON string for activation field");
}

int read_json_int_at(const std::string& json, std::size_t pos, int fallback) {
    if (pos == std::string::npos || pos >= json.size()) {
        return fallback;
    }

    const char* begin = json.c_str() + pos;
    char* end = nullptr;
    const long parsed = std::strtol(begin, &end, 10);

    if (end == begin) {
        return fallback;
    }

    return static_cast<int>(parsed);
}

std::vector<int> read_json_int_array_at(const std::string& json, std::size_t pos) {
    std::vector<int> values;

    if (pos == std::string::npos || pos >= json.size() || json[pos] != '[') {
        return values;
    }

    ++pos;

    while (pos < json.size()) {
        pos = skip_ws(json, pos);

        if (pos < json.size() && json[pos] == ']') {
            break;
        }

        const char* begin = json.c_str() + pos;
        char* end = nullptr;
        const long parsed = std::strtol(begin, &end, 10);

        if (end == begin) {
            throw std::invalid_argument("shape must contain only integers");
        }

        values.push_back(static_cast<int>(parsed));
        pos = static_cast<std::size_t>(end - json.c_str());
        pos = skip_ws(json, pos);

        if (pos < json.size() && json[pos] == ',') {
            ++pos;
            continue;
        }

        if (pos < json.size() && json[pos] == ']') {
            break;
        }

        throw std::invalid_argument("invalid integer array syntax for shape");
    }

    return values;
}

std::string extract_string(const std::string& json, const std::string& field, const std::string& fallback = "") {
    return read_json_string_at(json, field_value_pos(json, field), fallback);
}

int extract_int(const std::string& json, const std::string& field, int fallback = 0) {
    return read_json_int_at(json, field_value_pos(json, field), fallback);
}

std::vector<int> extract_int_array(const std::string& json, const std::string& field) {
    return read_json_int_array_at(json, field_value_pos(json, field));
}

std::string encode_int_array(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }

    out << "]";
    return out.str();
}

} // namespace

std::string json_escape(const std::string& value) {
    std::ostringstream out;

    for (const char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }

    return out.str();
}

ActivationRequest decode_activation_request(const std::string& json) {
    require_json_object(json);

    ActivationRequest request;

    request.session_id = extract_string(json, "session_id");
    request.request_id = extract_string(json, "request_id");
    request.model_id = extract_string(json, "model_id");

    request.stage_index = extract_int(json, "stage_index");
    request.node_name = extract_string(json, "node_name");
    request.role = extract_string(json, "role");

    request.layer_start = extract_int(json, "layer_start");
    request.layer_end = extract_int(json, "layer_end");
    request.decode_step = extract_int(json, "decode_step");

    request.shape = extract_int_array(json, "shape");
    request.dtype = extract_string(json, "dtype", "synthetic");

    request.payload = extract_string(json, "payload");

    request.bytes_in = extract_int(json, "bytes_in", static_cast<int>(request.payload.size()));
    request.transport = extract_string(json, "transport", "http");

    return request;
}

std::string encode_activation_response(const ActivationResponse& response) {
    std::ostringstream body;

    body
        << "{"
        << "\"session_id\":\"" << json_escape(response.session_id) << "\","
        << "\"request_id\":\"" << json_escape(response.request_id) << "\","
        << "\"model_id\":\"" << json_escape(response.model_id) << "\","

        << "\"stage_index\":" << response.stage_index << ","
        << "\"node_name\":\"" << json_escape(response.node_name) << "\","
        << "\"role\":\"" << json_escape(response.role) << "\","

        << "\"layer_start\":" << response.layer_start << ","
        << "\"layer_end\":" << response.layer_end << ","
        << "\"decode_step\":" << response.decode_step << ","

        << "\"shape\":" << encode_int_array(response.shape) << ","
        << "\"dtype\":\"" << json_escape(response.dtype) << "\","

        << "\"payload\":\"" << json_escape(response.payload) << "\","

        << "\"bytes_in\":" << response.bytes_in << ","
        << "\"bytes_out\":" << response.bytes_out << ","

        << "\"transport\":\"" << json_escape(response.transport) << "\","
        << "\"latency_ms\":" << response.latency_ms << ","

        << "\"trace\":{"
        << "\"stage_index\":" << response.trace.stage_index << ","
        << "\"node_name\":\"" << json_escape(response.trace.node_name) << "\","
        << "\"role\":\"" << json_escape(response.trace.role) << "\","
        << "\"layer_start\":" << response.trace.layer_start << ","
        << "\"layer_end\":" << response.trace.layer_end << ","
        << "\"bytes_in\":" << response.trace.bytes_in << ","
        << "\"bytes_out\":" << response.trace.bytes_out << ","
        << "\"transport\":\"" << json_escape(response.trace.transport) << "\","
        << "\"latency_ms\":" << response.trace.latency_ms
        << "}"
        << "}";

    return body.str();
}

} // namespace jetsonfabric::runtime::protocol