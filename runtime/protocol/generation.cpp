#include "protocol/generation.hpp"

#include <cctype>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace jetsonfabric::runtime::protocol {
namespace {

const nlohmann::json& require_field(const nlohmann::json& body, const char* field) {
    const auto value = body.find(field);
    if (value == body.end() || value->is_null()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    return *value;
}

std::string require_string(const nlohmann::json& body, const char* field) {
    const nlohmann::json& value = require_field(body, field);
    if (!value.is_string()) {
        throw std::invalid_argument(std::string(field) + " must be a string");
    }
    std::string parsed = value.get<std::string>();
    if (parsed.empty()) {
        throw std::invalid_argument(std::string(field) + " is required");
    }
    return parsed;
}

int require_int(const nlohmann::json& body, const char* field) {
    const nlohmann::json& value = require_field(body, field);
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }
    const std::int64_t parsed = value.get<std::int64_t>();
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(field) + " is outside int range");
    }
    return static_cast<int>(parsed);
}

std::uint64_t require_epoch(const nlohmann::json& body) {
    const nlohmann::json& value = require_field(body, "epoch");
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::invalid_argument("epoch must be a positive integer");
    }
    const std::int64_t parsed = value.get<std::int64_t>();
    if (parsed <= 0) {
        throw std::invalid_argument("epoch must be positive");
    }
    return static_cast<std::uint64_t>(parsed);
}

void validate_sha256(const std::string& value) {
    if (value.size() != 64) {
        throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
    }
    for (const unsigned char character : value) {
        if (!std::isxdigit(character)) {
            throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
        }
    }
}

deployment::DeploymentIdentity decode_deployment(const nlohmann::json& body) {
    deployment::DeploymentIdentity identity{
        .deployment_id = require_string(body, "deployment_id"),
        .epoch = require_epoch(body),
        .model_id = require_string(body, "model_id"),
        .model_sha256 = require_string(body, "model_sha256"),
    };
    validate_sha256(identity.model_sha256);
    return identity;
}

GenerationStage decode_stage(const nlohmann::json& body) {
    if (!body.is_object()) {
        throw std::invalid_argument("generation stage must be an object");
    }
    return GenerationStage{
        .stage_index = require_int(body, "stage_index"),
        .stage_count = require_int(body, "stage_count"),
        .node_id = require_string(body, "node_id"),
        .node_name = require_string(body, "node_name"),
        .api_url = require_string(body, "api_url"),
        .layer_start = require_int(body, "layer_start"),
        .layer_end = require_int(body, "layer_end"),
    };
}

void validate_stages(const std::vector<GenerationStage>& stages) {
    if (stages.empty() || stages.size() > 64) {
        throw std::invalid_argument("generation requires between 1 and 64 stages");
    }
    std::set<std::string> node_ids;
    int expected_start = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        const GenerationStage& stage = stages[index];
        if (stage.stage_index != static_cast<int>(index) ||
            stage.stage_count != static_cast<int>(stages.size())) {
            throw std::invalid_argument("generation stages have inconsistent indexes or counts");
        }
        if (!node_ids.insert(stage.node_id).second) {
            throw std::invalid_argument("generation stages must use distinct node IDs");
        }
        if (stage.api_url.rfind("http://", 0) != 0) {
            throw std::invalid_argument("generation stage api_url must use http://");
        }
        if (stage.layer_start != expected_start || stage.layer_end <= stage.layer_start) {
            throw std::invalid_argument("generation stages must contain contiguous non-empty layer ranges");
        }
        expected_start = stage.layer_end;
    }
}

} // namespace

GenerationRequest decode_generation_request(const std::string& body_text) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(body_text);
    } catch (const nlohmann::json::parse_error&) {
        throw std::invalid_argument("generation request body must be valid JSON");
    }
    if (!body.is_object()) {
        throw std::invalid_argument("generation request body must be an object");
    }

    GenerationRequest request;
    request.request_id = require_string(body, "request_id");
    request.session_id = require_string(body, "session_id");
    request.model_id = require_string(body, "model_id");
    request.prompt = require_string(body, "prompt");
    request.max_tokens = require_int(body, "max_tokens");
    if (request.max_tokens <= 0) request.max_tokens = 128;
    if (request.max_tokens > 1024) request.max_tokens = 1024;

    const auto deployment_value = body.find("deployment");
    if (deployment_value != body.end() && !deployment_value->is_null()) {
        if (!deployment_value->is_object()) {
            throw std::invalid_argument("deployment must be an object");
        }
        request.deployment = decode_deployment(*deployment_value);
        if (request.deployment->model_id != request.model_id) {
            throw std::invalid_argument("deployment model_id does not match generation model_id");
        }
    }

    const nlohmann::json& stages = require_field(body, "stages");
    if (!stages.is_array()) {
        throw std::invalid_argument("stages must be an array");
    }
    request.stages.reserve(stages.size());
    for (const nlohmann::json& stage : stages) {
        request.stages.push_back(decode_stage(stage));
    }
    validate_stages(request.stages);
    return request;
}

std::string encode_generation_token_event(std::uint32_t token, const std::string& text, int index) {
    return nlohmann::ordered_json{
        {"type", "token"},
        {"token", token},
        {"text", text},
        {"index", index},
    }.dump();
}

std::string encode_generation_done_event(
    const std::string& finish_reason,
    int prompt_tokens,
    int completion_tokens,
    const std::vector<std::uint32_t>& sampled_tokens,
    int stage_calls,
    int remote_stage_calls,
    std::int64_t bytes_in,
    std::int64_t bytes_out
) {
    return nlohmann::ordered_json{
        {"type", "done"},
        {"finish_reason", finish_reason},
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"sampled_tokens", sampled_tokens},
        {"stage_calls", stage_calls},
        {"remote_stage_calls", remote_stage_calls},
        {"bytes_in", bytes_in},
        {"bytes_out", bytes_out},
    }.dump();
}

std::string encode_generation_error_event(const std::string& code, const std::string& message) {
    return nlohmann::ordered_json{
        {"type", "error"},
        {"code", code},
        {"message", message},
    }.dump();
}

} // namespace jetsonfabric::runtime::protocol
