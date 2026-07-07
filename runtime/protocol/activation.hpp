#pragma once

#include <string>
#include <vector>

namespace jetsonfabric::runtime::protocol {

struct ActivationRequest {
    std::string session_id;
    std::string request_id;
    std::string model_id;

    int stage_index = 0;
    std::string node_name;
    std::string role;

    int layer_start = 0;
    int layer_end = 0;
    int decode_step = 0;

    // For now this describes the activation payload.
    // Later this should describe real hidden-state tensor bytes.
    std::vector<int> shape;
    std::string dtype = "synthetic";

    // Temporary JSON/string payload.
    // Later this becomes binary activation data or a binary frame reference.
    std::string payload;

    int bytes_in = 0;
    std::string transport = "http";

    int max_tokens = 128;
};

struct StageTrace {
    int stage_index = 0;
    std::string node_name;
    std::string role;

    int layer_start = 0;
    int layer_end = 0;

    int bytes_in = 0;
    int bytes_out = 0;

    std::string transport = "http";
    int latency_ms = 0;
};

struct ActivationResponse {
    std::string session_id;
    std::string request_id;
    std::string model_id;

    int stage_index = 0;
    std::string node_name;
    std::string role;

    int layer_start = 0;
    int layer_end = 0;
    int decode_step = 0;

    std::vector<int> shape;
    std::string dtype = "synthetic";

    std::string payload;

    int bytes_in = 0;
    int bytes_out = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;

    std::string transport = "http";
    int latency_ms = 0;

    StageTrace trace;
};

ActivationRequest decode_activation_request(const std::string& json);
std::string encode_activation_response(const ActivationResponse& response);

std::string json_escape(const std::string& value);

} // namespace jetsonfabric::runtime::protocol
