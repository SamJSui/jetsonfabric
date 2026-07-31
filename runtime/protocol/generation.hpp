#pragma once

#include "deployment/deployment.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::protocol {

inline constexpr const char* kGenerationContentType = "application/x-ndjson";

struct GenerationStage {
    int stage_index = 0;
    int stage_count = 1;
    std::string node_id;
    std::string node_name;
    std::string api_url;
    int layer_start = 0;
    int layer_end = 0;
};

struct GenerationRequest {
    std::string request_id;
    std::string session_id;
    std::string model_id;
    std::string prompt;
    int max_tokens = 128;
    std::optional<deployment::DeploymentIdentity> deployment;
    std::vector<GenerationStage> stages;
};

struct GenerationStageTiming {
    std::string phase;
    int stage_index = 0;
    std::string node_name;
    bool remote = false;
    int calls = 0;
    std::int64_t execution_us = 0;
    std::int64_t activation_decode_us = 0;
    std::int64_t activation_encode_us = 0;
    std::int64_t stage_total_us = 0;
    std::int64_t remote_call_us = 0;
    std::int64_t remote_overhead_us = 0;
    std::int64_t bytes_in = 0;
    std::int64_t bytes_out = 0;
};

GenerationRequest decode_generation_request(const std::string& body);
std::string encode_generation_token_event(
    std::uint32_t token,
    const std::string& text,
    int index
);
std::string encode_generation_done_event(
    const std::string& finish_reason,
    int prompt_tokens,
    int completion_tokens,
    const std::vector<std::uint32_t>& sampled_tokens,
    int stage_calls,
    int remote_stage_calls,
    std::int64_t bytes_in,
    std::int64_t bytes_out,
    const std::vector<GenerationStageTiming>& stage_timings
);
std::string encode_generation_error_event(const std::string& code, const std::string& message);

} // namespace jetsonfabric::runtime::protocol
