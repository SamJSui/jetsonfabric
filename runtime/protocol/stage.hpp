#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::protocol {

inline constexpr std::uint16_t kStageWireVersion = 1;
inline constexpr const char* kStageWireContentType = "application/vnd.jetsonfabric.stage.v1+octet-stream";
inline constexpr const char* kStageWireTransport = "http_binary_v1";

struct StageRequest {
    std::uint16_t protocol_version = kStageWireVersion;

    std::string session_id;
    std::string request_id;
    std::string model_id;

    std::string phase = "prefill";
    int decode_step = 0;

    int stage_index = 0;
    int stage_count = 1;
    std::string node_name;

    int layer_start = 0;
    int layer_end = 0;

    std::string payload_kind = "text";
    std::string encoding = "utf-8";
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string byte_order;
    std::string layout;
    std::vector<std::uint8_t> payload;

    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_crc32 = 0;
    std::string transport = kStageWireTransport;

    int max_tokens = 128;

    bool is_first_stage() const;
    bool is_last_stage() const;
};

struct StageResponse {
    std::uint16_t protocol_version = kStageWireVersion;

    std::string session_id;
    std::string request_id;
    std::string model_id;

    std::string phase = "prefill";
    int decode_step = 0;

    int stage_index = 0;
    int stage_count = 1;
    std::string node_name;

    int layer_start = 0;
    int layer_end = 0;

    std::string payload_kind = "text";
    std::string encoding = "utf-8";
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::string byte_order;
    std::string layout;
    std::vector<std::uint8_t> payload;

    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_crc32 = 0;
    std::string transport = kStageWireTransport;

    std::int64_t bytes_in = 0;
    std::int64_t bytes_out = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int latency_ms = 0;

    std::string error;
    std::string message;
};

StageRequest decode_stage_request(const std::string& frame);
std::string encode_stage_response(StageResponse response);
std::uint32_t payload_crc32(const std::vector<std::uint8_t>& payload);
std::string json_escape(const std::string& value);

} // namespace jetsonfabric::runtime::protocol
