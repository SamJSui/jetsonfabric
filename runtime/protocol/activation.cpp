#include "protocol/activation.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace jetsonfabric::runtime::protocol {
namespace {

nlohmann::json parse_json_object(const std::string& text) {
    try {
        nlohmann::json body = nlohmann::json::parse(text);

        if (!body.is_object()) {
            throw std::invalid_argument("activation request must be a JSON object");
        }

        return body;
    } catch (const nlohmann::json::parse_error& err) {
        throw std::invalid_argument(
            std::string("activation request must be valid JSON: ") + err.what()
        );
    }
}

const nlohmann::json* optional_field(const nlohmann::json& body, const char* field) {
    const auto it = body.find(field);

    if (it == body.end() || it->is_null()) {
        return nullptr;
    }

    return &(*it);
}

int json_int_value(const nlohmann::json& value, const char* field) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }

    long long parsed = 0;

    try {
        parsed = value.get<long long>();
    } catch (const nlohmann::json::exception&) {
        throw std::invalid_argument(std::string(field) + " is outside integer range");
    }

    if (
        parsed < static_cast<long long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long long>(std::numeric_limits<int>::max())
    ) {
        throw std::invalid_argument(std::string(field) + " is outside int range");
    }

    return static_cast<int>(parsed);
}

int extract_int(const nlohmann::json& body, const char* field, int fallback = 0) {
    const nlohmann::json* value = optional_field(body, field);

    if (value == nullptr) {
        return fallback;
    }

    return json_int_value(*value, field);
}

std::string extract_string(
    const nlohmann::json& body,
    const char* field,
    const std::string& fallback = ""
) {
    const nlohmann::json* value = optional_field(body, field);

    if (value == nullptr) {
        return fallback;
    }

    if (!value->is_string()) {
        throw std::invalid_argument(std::string(field) + " must be a string");
    }

    return value->get<std::string>();
}

std::vector<int> extract_int_array(const nlohmann::json& body, const char* field) {
    const nlohmann::json* value = optional_field(body, field);

    if (value == nullptr) {
        return {};
    }

    if (!value->is_array()) {
        throw std::invalid_argument(std::string(field) + " must be an array");
    }

    std::vector<int> values;
    values.reserve(value->size());

    for (const auto& item : *value) {
        values.push_back(json_int_value(item, field));
    }

    return values;
}

int normalize_max_tokens(int value) {
    if (value <= 0) {
        return 128;
    }

    if (value > 1024) {
        return 1024;
    }

    return value;
}

} // namespace

std::string json_escape(const std::string& value) {
    const std::string dumped = nlohmann::json(value).dump();

    if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
        return dumped.substr(1, dumped.size() - 2);
    }

    return dumped;
}

ActivationRequest decode_activation_request(const std::string& text) {
    const nlohmann::json body = parse_json_object(text);

    ActivationRequest request;

    request.session_id = extract_string(body, "session_id");
    request.request_id = extract_string(body, "request_id");
    request.model_id = extract_string(body, "model_id");

    request.stage_index = extract_int(body, "stage_index", 0);
    request.node_name = extract_string(body, "node_name");
    request.role = extract_string(body, "role");

    request.layer_start = extract_int(body, "layer_start", 0);
    request.layer_end = extract_int(body, "layer_end", 0);
    request.decode_step = extract_int(body, "decode_step", 0);

    request.shape = extract_int_array(body, "shape");
    request.dtype = extract_string(body, "dtype", "synthetic");
    request.payload = extract_string(body, "payload");

    request.bytes_in = extract_int(
        body,
        "bytes_in",
        static_cast<int>(request.payload.size())
    );

    request.transport = extract_string(body, "transport", "http");
    request.max_tokens = normalize_max_tokens(extract_int(body, "max_tokens", 128));

    return request;
}

std::string encode_activation_response(const ActivationResponse& response) {
    nlohmann::ordered_json trace;
    trace["stage_index"] = response.trace.stage_index;
    trace["node_name"] = response.trace.node_name;
    trace["role"] = response.trace.role;
    trace["layer_start"] = response.trace.layer_start;
    trace["layer_end"] = response.trace.layer_end;
    trace["bytes_in"] = response.trace.bytes_in;
    trace["bytes_out"] = response.trace.bytes_out;
    trace["transport"] = response.trace.transport;
    trace["latency_ms"] = response.trace.latency_ms;

    nlohmann::ordered_json body;
    body["session_id"] = response.session_id;
    body["request_id"] = response.request_id;
    body["model_id"] = response.model_id;

    body["stage_index"] = response.stage_index;
    body["node_name"] = response.node_name;
    body["role"] = response.role;

    body["layer_start"] = response.layer_start;
    body["layer_end"] = response.layer_end;
    body["decode_step"] = response.decode_step;

    body["shape"] = response.shape;
    body["dtype"] = response.dtype;
    body["payload"] = response.payload;

    body["bytes_in"] = response.bytes_in;
    body["bytes_out"] = response.bytes_out;

    body["transport"] = response.transport;
    body["latency_ms"] = response.latency_ms;
    body["trace"] = trace;

    return body.dump();
}

} // namespace jetsonfabric::runtime::protocol