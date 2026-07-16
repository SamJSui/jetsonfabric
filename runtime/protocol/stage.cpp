#include "protocol/stage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace jetsonfabric::runtime::protocol {
namespace {

constexpr std::size_t kHeaderSize = 20;
constexpr std::uint32_t kMaxMetadataBytes = 1U << 20;
constexpr std::uint64_t kMaxPayloadBytes = 512ULL << 20;
constexpr std::array<char, 4> kMagic{'J', 'F', 'S', 'T'};

std::uint16_t read_u16_be(const char* data) {
    return static_cast<std::uint16_t>((static_cast<unsigned char>(data[0]) << 8U) |
                                      static_cast<unsigned char>(data[1]));
}

std::uint32_t read_u32_be(const char* data) {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(data[0])) << 24U) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[1])) << 16U) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(data[2])) << 8U) |
           static_cast<std::uint32_t>(static_cast<unsigned char>(data[3]));
}

std::uint64_t read_u64_be(const char* data) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8U) | static_cast<unsigned char>(data[i]);
    }
    return value;
}

void append_u16_be(std::string& out, std::uint16_t value) {
    out.push_back(static_cast<char>((value >> 8U) & 0xffU));
    out.push_back(static_cast<char>(value & 0xffU));
}

void append_u32_be(std::string& out, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<char>((value >> static_cast<unsigned>(shift)) & 0xffU));
    }
}

void append_u64_be(std::string& out, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        out.push_back(static_cast<char>((value >> static_cast<unsigned>(shift)) & 0xffU));
    }
}

nlohmann::json parse_metadata(const std::string& text) {
    try {
        nlohmann::json body = nlohmann::json::parse(text);
        if (!body.is_object()) {
            throw std::invalid_argument("stagewire metadata must be a JSON object");
        }
        return body;
    } catch (const nlohmann::json::parse_error& err) {
        throw std::invalid_argument(std::string("stagewire metadata must be valid JSON: ") + err.what());
    }
}

const nlohmann::json* optional_field(const nlohmann::json& body, const char* field) {
    const auto it = body.find(field);
    if (it == body.end() || it->is_null()) {
        return nullptr;
    }
    return &(*it);
}

std::int64_t json_int64_value(const nlohmann::json& value, const char* field) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }
    try {
        return value.get<std::int64_t>();
    } catch (const nlohmann::json::exception&) {
        throw std::invalid_argument(std::string(field) + " is outside integer range");
    }
}

int extract_int(const nlohmann::json& body, const char* field, int fallback = 0) {
    const nlohmann::json* value = optional_field(body, field);
    if (value == nullptr) {
        return fallback;
    }
    const std::int64_t parsed = json_int64_value(*value, field);
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(field) + " is outside int range");
    }
    return static_cast<int>(parsed);
}

std::int64_t extract_int64(const nlohmann::json& body, const char* field, std::int64_t fallback = 0) {
    const nlohmann::json* value = optional_field(body, field);
    return value == nullptr ? fallback : json_int64_value(*value, field);
}

std::uint32_t extract_u32(const nlohmann::json& body, const char* field, std::uint32_t fallback = 0) {
    const std::int64_t parsed = extract_int64(body, field, fallback);
    if (parsed < 0 || static_cast<std::uint64_t>(parsed) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(field) + " is outside uint32 range");
    }
    return static_cast<std::uint32_t>(parsed);
}

std::string extract_string(const nlohmann::json& body, const char* field, const std::string& fallback = "") {
    const nlohmann::json* value = optional_field(body, field);
    if (value == nullptr) {
        return fallback;
    }
    if (!value->is_string()) {
        throw std::invalid_argument(std::string(field) + " must be a string");
    }
    return value->get<std::string>();
}

std::vector<std::int64_t> extract_shape(const nlohmann::json& body) {
    const nlohmann::json* value = optional_field(body, "shape");
    if (value == nullptr) {
        return {};
    }
    if (!value->is_array()) {
        throw std::invalid_argument("shape must be an array");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(value->size());
    for (const auto& item : *value) {
        shape.push_back(json_int64_value(item, "shape"));
    }
    return shape;
}

std::uint64_t dtype_width(const std::string& dtype) {
    if (dtype == "u8" || dtype == "i8") return 1;
    if (dtype == "f16" || dtype == "bf16") return 2;
    if (dtype == "i32" || dtype == "u32" || dtype == "f32") return 4;
    if (dtype == "i64" || dtype == "u64" || dtype == "f64") return 8;
    throw std::invalid_argument("unsupported dtype: " + dtype);
}

void validate_tensor_metadata(
    const std::string& kind,
    const std::string& encoding,
    const std::string& dtype,
    const std::vector<std::int64_t>& shape,
    const std::string& byte_order,
    const std::string& layout,
    std::size_t payload_size
) {
    if (kind == "text") {
        if (encoding != "utf-8") {
            throw std::invalid_argument("text payload encoding must be utf-8");
        }
        if (!dtype.empty() || !shape.empty() || !byte_order.empty() || !layout.empty()) {
            throw std::invalid_argument("text payload must not declare tensor metadata");
        }
        return;
    }
    if (kind != "tokens" && kind != "activation" && kind != "sampled_token") {
        throw std::invalid_argument("invalid payload_kind: " + kind);
    }
    if (dtype.empty() || shape.empty()) {
        throw std::invalid_argument("tensor payload requires dtype and shape");
    }
    if (byte_order != "little" || layout != "row_major") {
        throw std::invalid_argument("tensor payload requires little byte order and row_major layout");
    }
    std::uint64_t count = 1;
    for (const std::int64_t dim : shape) {
        if (dim <= 0 || count > kMaxPayloadBytes / static_cast<std::uint64_t>(dim)) {
            throw std::invalid_argument("invalid or oversized tensor shape");
        }
        count *= static_cast<std::uint64_t>(dim);
    }
    const std::uint64_t width = dtype_width(dtype);
    if (count > kMaxPayloadBytes / width || count * width != payload_size) {
        throw std::invalid_argument("tensor shape and dtype do not match payload length");
    }
}

struct DecodedFrame {
    nlohmann::json metadata;
    std::vector<std::uint8_t> payload;
};

DecodedFrame decode_frame(const std::string& frame) {
    if (frame.size() < kHeaderSize) {
        throw std::invalid_argument("truncated stagewire header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), frame.begin())) {
        throw std::invalid_argument("invalid stagewire magic");
    }
    const std::uint16_t version = read_u16_be(frame.data() + 4);
    if (version != kStageWireVersion) {
        throw std::invalid_argument("unsupported stagewire version: " + std::to_string(version));
    }
    if (read_u16_be(frame.data() + 6) != 0) {
        throw std::invalid_argument("stagewire flags must be zero");
    }
    const std::uint32_t metadata_length = read_u32_be(frame.data() + 8);
    const std::uint64_t payload_length = read_u64_be(frame.data() + 12);
    if (metadata_length == 0 || metadata_length > kMaxMetadataBytes || payload_length > kMaxPayloadBytes) {
        throw std::invalid_argument("stagewire frame is too large");
    }
    const std::uint64_t expected = kHeaderSize + static_cast<std::uint64_t>(metadata_length) + payload_length;
    if (expected != frame.size()) {
        throw std::invalid_argument(expected > frame.size() ? "truncated stagewire frame" : "stagewire frame has trailing data");
    }
    const std::size_t metadata_start = kHeaderSize;
    const std::size_t payload_start = metadata_start + metadata_length;
    std::string metadata_text = frame.substr(metadata_start, metadata_length);
    std::vector<std::uint8_t> payload(payload_length);
    if (payload_length > 0) {
        std::memcpy(payload.data(), frame.data() + payload_start, static_cast<std::size_t>(payload_length));
    }
    nlohmann::json metadata = parse_metadata(metadata_text);
    if (extract_int(metadata, "protocol_version") != version) {
        throw std::invalid_argument("metadata protocol_version does not match frame version");
    }
    if (extract_int64(metadata, "payload_bytes") != static_cast<std::int64_t>(payload.size())) {
        throw std::invalid_argument("payload_bytes does not match frame payload length");
    }
    if (extract_u32(metadata, "payload_crc32") != payload_crc32(payload)) {
        throw std::invalid_argument("stagewire payload checksum mismatch");
    }
    if (extract_string(metadata, "transport") != kStageWireTransport) {
        throw std::invalid_argument("invalid stagewire transport");
    }
    return DecodedFrame{std::move(metadata), std::move(payload)};
}

std::string encode_frame(nlohmann::ordered_json metadata, const std::vector<std::uint8_t>& payload) {
    metadata["protocol_version"] = kStageWireVersion;
    metadata["payload_bytes"] = payload.size();
    metadata["payload_crc32"] = payload_crc32(payload);
    metadata["transport"] = kStageWireTransport;
    const std::string metadata_text = metadata.dump();
    if (metadata_text.size() > kMaxMetadataBytes || payload.size() > kMaxPayloadBytes) {
        throw std::invalid_argument("stagewire frame is too large");
    }
    std::string out;
    out.reserve(kHeaderSize + metadata_text.size() + payload.size());
    out.append(kMagic.data(), kMagic.size());
    append_u16_be(out, kStageWireVersion);
    append_u16_be(out, 0);
    append_u32_be(out, static_cast<std::uint32_t>(metadata_text.size()));
    append_u64_be(out, payload.size());
    out.append(metadata_text);
    if (!payload.empty()) {
        out.append(reinterpret_cast<const char*>(payload.data()), payload.size());
    }
    return out;
}

int normalize_max_tokens(int value) {
    if (value <= 0) return 128;
    return value > 1024 ? 1024 : value;
}

} // namespace

bool StageRequest::is_first_stage() const {
    return stage_count > 0 && stage_index == 0;
}

bool StageRequest::is_last_stage() const {
    return stage_count > 0 && stage_index == stage_count - 1;
}

std::uint32_t payload_crc32(const std::vector<std::uint8_t>& payload) {
    std::uint32_t crc = 0xffffffffU;
    for (const std::uint8_t value : payload) {
        crc ^= value;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0U - (crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

std::string json_escape(const std::string& value) {
    const std::string dumped = nlohmann::json(value).dump();
    return dumped.size() >= 2 ? dumped.substr(1, dumped.size() - 2) : dumped;
}

StageRequest decode_stage_request(const std::string& frame) {
    DecodedFrame decoded = decode_frame(frame);
    const nlohmann::json& body = decoded.metadata;
    StageRequest request;
    request.protocol_version = static_cast<std::uint16_t>(extract_int(body, "protocol_version"));
    request.session_id = extract_string(body, "session_id");
    request.request_id = extract_string(body, "request_id");
    request.model_id = extract_string(body, "model_id");
    request.phase = extract_string(body, "phase", extract_int(body, "decode_step") == 0 ? "prefill" : "decode");
    request.decode_step = extract_int(body, "decode_step");
    request.stage_index = extract_int(body, "stage_index");
    request.stage_count = extract_int(body, "stage_count", 1);
    request.node_name = extract_string(body, "node_name");
    request.layer_start = extract_int(body, "layer_start");
    request.layer_end = extract_int(body, "layer_end");
    request.payload_kind = extract_string(body, "payload_kind");
    request.encoding = extract_string(body, "encoding");
    request.dtype = extract_string(body, "dtype");
    request.shape = extract_shape(body);
    request.byte_order = extract_string(body, "byte_order");
    request.layout = extract_string(body, "layout");
    request.payload = std::move(decoded.payload);
    request.payload_bytes = request.payload.size();
    request.payload_crc32 = payload_crc32(request.payload);
    request.transport = extract_string(body, "transport");
    request.max_tokens = normalize_max_tokens(extract_int(body, "max_tokens", 128));
    validate_tensor_metadata(request.payload_kind, request.encoding, request.dtype, request.shape,
                             request.byte_order, request.layout, request.payload.size());
    return request;
}

std::string encode_stage_response(StageResponse response) {
    response.protocol_version = kStageWireVersion;
    response.payload_bytes = response.payload.size();
    response.payload_crc32 = payload_crc32(response.payload);
    response.transport = kStageWireTransport;
    if (response.error.empty()) {
        validate_tensor_metadata(response.payload_kind, response.encoding, response.dtype, response.shape,
                                 response.byte_order, response.layout, response.payload.size());
    }

    nlohmann::ordered_json body;
    body["protocol_version"] = response.protocol_version;
    body["session_id"] = response.session_id;
    body["request_id"] = response.request_id;
    body["model_id"] = response.model_id;
    body["phase"] = response.phase;
    body["decode_step"] = response.decode_step;
    body["stage_index"] = response.stage_index;
    body["stage_count"] = response.stage_count;
    body["node_name"] = response.node_name;
    body["layer_start"] = response.layer_start;
    body["layer_end"] = response.layer_end;
    body["payload_kind"] = response.payload_kind;
    if (!response.encoding.empty()) body["encoding"] = response.encoding;
    if (!response.dtype.empty()) body["dtype"] = response.dtype;
    if (!response.shape.empty()) body["shape"] = response.shape;
    if (!response.byte_order.empty()) body["byte_order"] = response.byte_order;
    if (!response.layout.empty()) body["layout"] = response.layout;
    body["payload_bytes"] = response.payload_bytes;
    body["payload_crc32"] = response.payload_crc32;
    body["transport"] = response.transport;
    if (response.bytes_in != 0) body["bytes_in"] = response.bytes_in;
    if (response.bytes_out != 0) body["bytes_out"] = response.bytes_out;
    if (response.prompt_tokens != 0) body["prompt_tokens"] = response.prompt_tokens;
    if (response.completion_tokens != 0) body["completion_tokens"] = response.completion_tokens;
    if (response.latency_ms != 0) body["latency_ms"] = response.latency_ms;
    if (!response.error.empty()) body["error"] = response.error;
    if (!response.message.empty()) body["message"] = response.message;
    return encode_frame(std::move(body), response.payload);
}

} // namespace jetsonfabric::runtime::protocol
