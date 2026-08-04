#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::protocol {

inline constexpr std::uint16_t kStageWireVersion = 2;
inline constexpr const char* kStageWireContentType = "application/vnd.jetsonfabric.stage.v2+octet-stream";
inline constexpr const char* kStageWireTransport = "http_binary_v1";
inline constexpr const char* kDirectStageTransport = "http_direct_v1";
inline constexpr int kMaxStageTokens = 1024;

struct StageRequest {
    std::uint16_t protocol_version = kStageWireVersion;

    std::string session_id;
    std::string request_id;
    std::string model_id;
    std::string deployment_id;
    std::uint64_t deployment_epoch = 0;
    std::string model_sha256;

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
    int rollback_tokens = 0;

    bool is_first_stage() const;
    bool is_last_stage() const;
};

struct StageResponse {
    std::uint16_t protocol_version = kStageWireVersion;
    std::string operation = "execute";

    std::string session_id;
    std::string request_id;
    std::string model_id;
    std::string deployment_id;
    std::uint64_t deployment_epoch = 0;
    std::string model_sha256;

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
    std::vector<std::uint32_t> prompt_token_ids;
    int completion_tokens = 0;
    int execution_batch_size = 1;
    int verification_width = 1;
    int latency_ms = 0;
    std::int64_t execution_us = 0;
    std::int64_t activation_decode_us = 0;
    std::int64_t activation_encode_us = 0;
    std::int64_t stage_total_us = 0;

    std::string error;
    std::string message;
    std::vector<std::uint32_t> token_text_offsets;
    std::vector<std::uint8_t> token_eog;
};

struct EncodedStageFrameView {
    std::string prefix;
    std::span<const std::uint8_t> payload;

    std::size_t size() const noexcept { return prefix.size() + payload.size(); }
    std::string flatten() const;
};

struct EncodedStageFrame {
    std::string prefix;
    std::vector<std::uint8_t> payload;

    std::size_t size() const noexcept { return prefix.size() + payload.size(); }
    std::string flatten() const;
};

StageRequest decode_stage_request(const std::string& frame);
EncodedStageFrameView encode_stage_request_frame(
    const StageRequest& request,
    const std::string& operation
);
std::string encode_stage_request(
    const StageRequest& request,
    const std::string& operation
);
StageResponse decode_stage_response(const std::string& frame);
EncodedStageFrame encode_stage_response_frame(StageResponse response);
std::string encode_stage_response(const StageResponse& response);
std::uint32_t payload_crc32(const std::vector<std::uint8_t>& payload);
std::string json_escape(const std::string& value);

} // namespace jetsonfabric::runtime::protocol
