#include "pipeline_parallel/generation_runner.hpp"

#include "inference/stage.hpp"

#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

GenerationResult generation_error(std::string status, std::string code, std::string message) {
    GenerationResult result;
    result.status = std::move(status);
    result.error_code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

void write_u32_le(std::vector<std::uint8_t>& payload, std::uint32_t value) {
    payload = {
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8U) & 0xffU),
        static_cast<std::uint8_t>((value >> 16U) & 0xffU),
        static_cast<std::uint8_t>((value >> 24U) & 0xffU),
    };
}

std::uint32_t read_sampled_token(const protocol::StageResponse& response) {
    if (response.payload_kind != "sampled_token" || response.payload.size() != 4U) {
        throw std::invalid_argument("final stage did not return one sampled_token");
    }
    return static_cast<std::uint32_t>(response.payload[0]) |
           (static_cast<std::uint32_t>(response.payload[1]) << 8U) |
           (static_cast<std::uint32_t>(response.payload[2]) << 16U) |
           (static_cast<std::uint32_t>(response.payload[3]) << 24U);
}

protocol::StageRequest initial_request(
    const protocol::GenerationRequest& request,
    const std::string& phase,
    int decode_step,
    std::uint32_t previous_token
) {
    protocol::StageRequest stage_request;
    stage_request.session_id = request.session_id;
    stage_request.request_id = request.request_id + "-" + phase + "-" + std::to_string(decode_step);
    stage_request.model_id = request.model_id;
    if (request.deployment.has_value()) {
        stage_request.deployment_id = request.deployment->deployment_id;
        stage_request.deployment_epoch = request.deployment->epoch;
        stage_request.model_sha256 = request.deployment->model_sha256;
    }
    stage_request.phase = phase;
    stage_request.decode_step = decode_step;
    stage_request.max_tokens = request.max_tokens;
    if (phase == "prefill") {
        stage_request.payload_kind = "text";
        stage_request.encoding = "utf-8";
        stage_request.payload.assign(request.prompt.begin(), request.prompt.end());
    } else {
        stage_request.payload_kind = "sampled_token";
        stage_request.encoding.clear();
        stage_request.dtype = "u32";
        stage_request.shape = {1};
        stage_request.byte_order = "little";
        stage_request.layout = "row_major";
        write_u32_le(stage_request.payload, previous_token);
    }
    return stage_request;
}

void apply_stage(const protocol::GenerationStage& stage, protocol::StageRequest& request) {
    request.request_id += "-stage-" + std::to_string(stage.stage_index);
    request.stage_index = stage.stage_index;
    request.stage_count = stage.stage_count;
    request.node_name = stage.node_name;
    request.layer_start = stage.layer_start;
    request.layer_end = stage.layer_end;
}

void apply_response(protocol::StageRequest& request, const protocol::StageResponse& response) {
    request.payload_kind = response.payload_kind;
    request.encoding = response.encoding;
    request.dtype = response.dtype;
    request.shape = response.shape;
    request.byte_order = response.byte_order;
    request.layout = response.layout;
    request.payload = response.payload;
}

std::string validate_response_identity(
    const protocol::StageRequest& request,
    const protocol::StageResponse& response
) {
    if (response.session_id != request.session_id ||
        response.request_id != request.request_id ||
        response.model_id != request.model_id ||
        response.deployment_id != request.deployment_id ||
        response.deployment_epoch != request.deployment_epoch ||
        response.model_sha256 != request.model_sha256 ||
        response.phase != request.phase ||
        response.decode_step != request.decode_step ||
        response.stage_index != request.stage_index ||
        response.stage_count != request.stage_count ||
        response.node_name != request.node_name ||
        response.layer_start != request.layer_start ||
        response.layer_end != request.layer_end) {
        return "stage response identity does not match its request";
    }
    return "";
}

std::string validate_response(
    const protocol::StageRequest& request,
    const protocol::StageResponse& response
) {
    const std::string identity_error = validate_response_identity(request, response);
    if (!identity_error.empty()) return identity_error;
    try {
        const inference::PayloadKind actual = inference::parse_payload_kind(response.payload_kind);
        const inference::PayloadKind expected = inference::expected_output(
            inference::parse_phase(request.phase),
            inference::StagePosition{.index = request.stage_index, .count = request.stage_count}
        );
        if (actual != expected) {
            return "stage response violates the pipeline payload transition";
        }
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

struct PassResult {
    bool ok = false;
    GenerationResult error;
    protocol::StageResponse final_response;
};

PassResult run_pass(
    const protocol::GenerationRequest& generation,
    const StageInvoker& invoke_stage,
    protocol::StageRequest request,
    GenerationResult& result
) {
    protocol::StageResponse final_response;
    const std::string pass_request_id = request.request_id;
    for (const protocol::GenerationStage& stage : generation.stages) {
        request.request_id = pass_request_id;
        apply_stage(stage, request);
        StageRunResult stage_result = invoke_stage(stage, request, StageOperation::Execute);
        ++result.stage_calls;
        if (stage.stage_index != 0) ++result.remote_stage_calls;
        if (!stage_result.ok) {
            std::ostringstream message;
            message << "stage " << stage.stage_index << ": " << stage_result.error_message;
            return PassResult{false, generation_error(
                stage_result.status,
                stage_result.error_code.empty() ? "generation_stage_failed" : stage_result.error_code,
                message.str()
            ), {}};
        }
        const std::string response_error = validate_response(request, stage_result.response);
        if (!response_error.empty()) {
            return PassResult{false, generation_error(
                "502 Bad Gateway", "invalid_stage_response", response_error
            ), {}};
        }
        result.prompt_tokens += stage_result.response.prompt_tokens;
        result.completion_tokens += stage_result.response.completion_tokens;
        result.bytes_in += stage_result.response.bytes_in;
        result.bytes_out += stage_result.response.bytes_out;
        final_response = stage_result.response;
        apply_response(request, stage_result.response);
    }
    return PassResult{true, {}, std::move(final_response)};
}

std::string close_sessions(const protocol::GenerationRequest& generation, const StageInvoker& invoke_stage) {
    std::string first_error;
    for (const protocol::GenerationStage& stage : generation.stages) {
        protocol::StageRequest request;
        request.session_id = generation.session_id;
        request.request_id = generation.request_id + "-close-stage-" + std::to_string(stage.stage_index);
        request.model_id = generation.model_id;
        if (generation.deployment.has_value()) {
            request.deployment_id = generation.deployment->deployment_id;
            request.deployment_epoch = generation.deployment->epoch;
            request.model_sha256 = generation.deployment->model_sha256;
        }
        request.phase = "prefill";
        request.stage_index = stage.stage_index;
        request.stage_count = stage.stage_count;
        request.node_name = stage.node_name;
        request.layer_start = stage.layer_start;
        request.layer_end = stage.layer_end;
        request.payload_kind = "text";
        request.encoding = "utf-8";
        request.max_tokens = 1;
        StageRunResult close = invoke_stage(stage, request, StageOperation::CloseSession);
        if (!close.ok && first_error.empty()) {
            first_error = "stage " + std::to_string(stage.stage_index) + ": " + close.error_message;
        } else if (close.ok) {
            const std::string response_error = validate_response_identity(request, close.response);
            if (!response_error.empty() && first_error.empty()) {
                first_error = "stage " + std::to_string(stage.stage_index) + ": " + response_error;
            }
        }
    }
    return first_error;
}

} // namespace

GenerationRunner::GenerationRunner(StageInvoker invoke_stage)
    : invoke_stage_(std::move(invoke_stage)) {
    if (!invoke_stage_) {
        throw std::invalid_argument("generation runner requires a stage invoker");
    }
}

GenerationResult GenerationRunner::run(
    const protocol::GenerationRequest& request,
    const TokenSink& sink
) const {
    if (!sink) {
        return generation_error("500 Internal Server Error", "generation_sink_missing", "token sink is required");
    }

    GenerationResult result;
    bool end_of_generation = false;
    std::uint32_t previous_token = 0;
    for (int pass = 0; pass < request.max_tokens && !end_of_generation; ++pass) {
        const std::string phase = pass == 0 ? "prefill" : "decode";
        protocol::StageRequest stage_request = initial_request(request, phase, pass, previous_token);
        PassResult pass_result = run_pass(request, invoke_stage_, std::move(stage_request), result);
        if (!pass_result.ok) {
            const std::string cleanup_error = close_sessions(request, invoke_stage_);
            if (!cleanup_error.empty()) pass_result.error.error_message += "; cleanup: " + cleanup_error;
            return pass_result.error;
        }

        std::uint32_t sampled_token = 0;
        try {
            sampled_token = read_sampled_token(pass_result.final_response);
        } catch (const std::exception& error) {
            const std::string cleanup_error = close_sessions(request, invoke_stage_);
            std::string message = error.what();
            if (!cleanup_error.empty()) message += "; cleanup: " + cleanup_error;
            return generation_error("502 Bad Gateway", "invalid_sampled_token", message);
        }
        end_of_generation = pass_result.final_response.completion_tokens == 0;
        if (end_of_generation) continue;

        previous_token = sampled_token;
        const int token_index = static_cast<int>(result.sampled_tokens.size());
        result.sampled_tokens.push_back(sampled_token);
        if (!sink(GenerationToken{sampled_token, pass_result.final_response.message, token_index})) {
            const std::string cleanup_error = close_sessions(request, invoke_stage_);
            std::string message = "generation token sink canceled the request";
            if (!cleanup_error.empty()) message += "; cleanup: " + cleanup_error;
            return generation_error("499 Client Closed Request", "generation_canceled", message);
        }
    }

    const std::string cleanup_error = close_sessions(request, invoke_stage_);
    if (!cleanup_error.empty()) {
        return generation_error("502 Bad Gateway", "generation_cleanup_failed", cleanup_error);
    }
    result.ok = true;
    result.status = "200 OK";
    result.finish_reason = end_of_generation ? "stop" : "length";
    return result;
}

} // namespace jetsonfabric::runtime::pipeline_parallel
