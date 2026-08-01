#include "pipeline_parallel/stage_worker.hpp"

#include "inference/stage.hpp"

#include <chrono>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

StageRunResult error_result(
    const std::string& status,
    const std::string& code,
    const std::string& message
) {
    StageRunResult result;
    result.ok = false;
    result.status = status;
    result.error_code = code;
    result.error_message = message;
    return result;
}

StageRunResult bad_request(const std::string& code, const std::string& message) {
    return error_result("400 Bad Request", code, message);
}

std::int64_t elapsed_us(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
}

int elapsed_ms(std::int64_t elapsed_microseconds) {
    return static_cast<int>(elapsed_microseconds / 1000);
}

struct StageTimings {
    std::int64_t execution_us = 0;
    std::int64_t activation_decode_us = 0;
    std::int64_t activation_encode_us = 0;
    std::int64_t stage_total_us = 0;
};

inference::Payload to_inference_payload(const protocol::StageRequest& request) {
    return inference::Payload{
        .kind = inference::parse_payload_kind(request.payload_kind),
        .encoding = request.encoding,
        .tensor = inference::TensorDescriptor{
            .dtype = request.dtype,
            .shape = request.shape,
            .byte_order = request.byte_order,
            .layout = request.layout,
        },
        .bytes = request.payload,
    };
}

inference::StageInput to_stage_input(const protocol::StageRequest& request) {
    return inference::StageInput{
        .session_id = request.session_id,
        .request_id = request.request_id,
        .model_id = request.model_id,
        .phase = inference::parse_phase(request.phase),
        .decode_step = request.decode_step,
        .position = inference::StagePosition{
            .index = request.stage_index,
            .count = request.stage_count,
        },
        .layers = inference::LayerRange{
            .start = request.layer_start,
            .end = request.layer_end,
        },
        .payload = to_inference_payload(request),
        .max_tokens = request.max_tokens,
    };
}

protocol::StageResponse base_response(const protocol::StageRequest& request) {
    protocol::StageResponse response;
    response.session_id = request.session_id;
    response.request_id = request.request_id;
    response.model_id = request.model_id;
    response.deployment_id = request.deployment_id;
    response.deployment_epoch = request.deployment_epoch;
    response.model_sha256 = request.model_sha256;
    response.phase = request.phase;
    response.decode_step = request.decode_step;
    response.stage_index = request.stage_index;
    response.stage_count = request.stage_count;
    response.node_name = request.node_name;
    response.layer_start = request.layer_start;
    response.layer_end = request.layer_end;
    return response;
}

protocol::StageResponse to_stage_response(
    const protocol::StageRequest& request,
    const inference::StageOutput& output,
    const StageTimings& timings
) {
    protocol::StageResponse response = base_response(request);
    response.payload_kind = inference::to_string(output.payload.kind);
    response.encoding = output.payload.encoding;
    response.dtype = output.payload.tensor.dtype;
    response.shape = output.payload.tensor.shape;
    response.byte_order = output.payload.tensor.byte_order;
    response.layout = output.payload.tensor.layout;
    response.payload = output.payload.bytes;
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    response.bytes_out = static_cast<std::int64_t>(response.payload.size());
    response.prompt_tokens = output.prompt_tokens;
    response.prompt_token_ids = output.prompt_token_ids;
    response.completion_tokens = output.completion_tokens;
    response.execution_batch_size = output.execution_batch_size;
    response.verification_width = output.verification_width;
    response.latency_ms = elapsed_ms(timings.execution_us);
    response.execution_us = timings.execution_us;
    response.activation_decode_us = timings.activation_decode_us;
    response.activation_encode_us = timings.activation_encode_us;
    response.stage_total_us = timings.stage_total_us;
    response.message = output.token_text;
    response.token_text_offsets = output.token_text_offsets;
    response.token_eog = output.token_eog;
    return response;
}

protocol::StageResponse close_response(const protocol::StageRequest& request, std::int64_t duration_us) {
    protocol::StageResponse response = base_response(request);
    response.payload_kind = "text";
    response.encoding = "utf-8";
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    response.latency_ms = elapsed_ms(duration_us);
    response.stage_total_us = duration_us;
    return response;
}

} // namespace

StageWorker::StageWorker(
    std::string node_name,
    std::string model_id,
    StageAssignment assignment,
    const LayerExecutor& layer_executor,
    std::shared_ptr<const activation::ActivationCodec> activation_codec
)
    : node_name_(std::move(node_name)),
      model_id_(std::move(model_id)),
      assignment_(assignment),
      layer_executor_(layer_executor),
      activation_codec_(std::move(activation_codec)) {
    if (!activation_codec_) {
        throw std::invalid_argument("stage worker requires an activation codec");
    }
}

StageRunResult StageWorker::run(const protocol::StageRequest& request) const {
    const std::string assignment_error = validate_stage_assignment(assignment_);
    if (!assignment_error.empty()) {
        return bad_request("invalid_stage_assignment", assignment_error);
    }

    const std::string request_error = validate_request(request);
    if (!request_error.empty()) {
        return bad_request("invalid_stage_request", request_error);
    }

    const auto stage_start = std::chrono::steady_clock::now();
    StageTimings timings;
    inference::StageInput input;
    try {
        input = to_stage_input(request);
    } catch (const std::exception& error) {
        return bad_request("invalid_stage_input", error.what());
    }

    const std::string wire_payload_error = inference::validate_payload(input.payload);
    if (!wire_payload_error.empty()) {
        return bad_request("invalid_stage_input", wire_payload_error);
    }
    if (input.payload.kind == inference::PayloadKind::Activation) {
        const auto decode_start = std::chrono::steady_clock::now();
        try {
            input.payload = activation_codec_->decode(std::move(input.payload));
        } catch (const std::exception& error) {
            return bad_request("activation_decode_failed", error.what());
        }
        timings.activation_decode_us = elapsed_us(decode_start);
    }

    const std::string input_error = inference::validate_stage_input(input);
    if (!input_error.empty()) {
        return bad_request("invalid_stage_input", input_error);
    }

    const auto execution_start = std::chrono::steady_clock::now();
    inference::ExecutionResult execution = layer_executor_.execute(input);
    timings.execution_us = elapsed_us(execution_start);
    if (!execution.ok) {
        const std::string status = execution.error.kind == inference::ErrorKind::InvalidInput
            ? "400 Bad Request"
            : "502 Bad Gateway";
        return error_result(status, execution.error.code, execution.error.message);
    }

    const std::string output_error = inference::validate_payload(execution.output.payload);
    if (!output_error.empty()) {
        return error_result("502 Bad Gateway", "invalid_stage_output", output_error);
    }

    if (execution.output.payload.kind == inference::PayloadKind::Activation) {
        const auto encode_start = std::chrono::steady_clock::now();
        try {
            execution.output.payload =
                activation_codec_->encode(std::move(execution.output.payload));
        } catch (const std::exception& error) {
            return error_result(
                "502 Bad Gateway",
                "activation_encode_failed",
                error.what()
            );
        }
        timings.activation_encode_us = elapsed_us(encode_start);
        const std::string encoded_output_error =
            inference::validate_payload(execution.output.payload);
        if (!encoded_output_error.empty()) {
            return error_result(
                "502 Bad Gateway",
                "invalid_stage_output",
                encoded_output_error
            );
        }
    }

    StageRunResult result;
    result.ok = true;
    result.status = "200 OK";
    timings.stage_total_us = elapsed_us(stage_start);
    result.response = to_stage_response(request, execution.output, timings);
    return result;
}

StageRunResult StageWorker::close_session(const protocol::StageRequest& request) const {
    const std::string assignment_error = validate_stage_assignment(assignment_);
    if (!assignment_error.empty()) {
        return bad_request("invalid_stage_assignment", assignment_error);
    }
    const std::string request_error = validate_request(request);
    if (!request_error.empty()) {
        return bad_request("invalid_stage_request", request_error);
    }

    const auto start = std::chrono::steady_clock::now();
    try {
        layer_executor_.close_session(request.session_id);
    } catch (const std::exception& error) {
        return error_result("502 Bad Gateway", "stage_session_close_failed", error.what());
    }

    StageRunResult result;
    result.ok = true;
    result.status = "200 OK";
    result.response = close_response(request, elapsed_us(start));
    return result;
}

StageRunResult StageWorker::rollback_session(const protocol::StageRequest& request) const {
    const std::string assignment_error = validate_stage_assignment(assignment_);
    if (!assignment_error.empty()) {
        return bad_request("invalid_stage_assignment", assignment_error);
    }
    const std::string request_error = validate_request(request);
    if (!request_error.empty()) {
        return bad_request("invalid_stage_request", request_error);
    }
    if (request.rollback_tokens <= 0) {
        return bad_request("invalid_rollback", "rollback_tokens must be greater than zero");
    }

    const auto start = std::chrono::steady_clock::now();
    try {
        layer_executor_.rollback_session(request.session_id, request.rollback_tokens);
    } catch (const std::invalid_argument& error) {
        return bad_request("stage_session_rollback_rejected", error.what());
    } catch (const std::exception& error) {
        return error_result("502 Bad Gateway", "stage_session_rollback_failed", error.what());
    }

    StageRunResult result;
    result.ok = true;
    result.status = "200 OK";
    result.response = close_response(request, elapsed_us(start));
    return result;
}

std::string StageWorker::validate_request(const protocol::StageRequest& request) const {
    if (request.session_id.empty()) {
        return "session_id is required";
    }
    if (request.request_id.empty()) {
        return "request_id is required";
    }
    if (request.model_id.empty()) {
        return "model_id is required";
    }
    if (request.model_id != model_id_) {
        return "request model_id " + request.model_id + " does not match runtime model_id " + model_id_;
    }
    if (request.stage_index != assignment_.stage_index) {
        std::ostringstream message;
        message << "request stage_index " << request.stage_index
                << " does not match runtime stage_index " << assignment_.stage_index;
        return message.str();
    }
    if (request.stage_count != assignment_.stage_count) {
        std::ostringstream message;
        message << "request stage_count " << request.stage_count
                << " does not match runtime stage_count " << assignment_.stage_count;
        return message.str();
    }
    if (request.layer_start != assignment_.layer_start || request.layer_end != assignment_.layer_end) {
        std::ostringstream message;
        message << "request layer range [" << request.layer_start << ':' << request.layer_end
                << "] does not match runtime assignment [" << assignment_.layer_start
                << ':' << assignment_.layer_end << ']';
        return message.str();
    }
    if (request.node_name != node_name_) {
        std::ostringstream message;
        message << "request node_name " << request.node_name
                << " does not match runtime node_name " << node_name_;
        return message.str();
    }
    if (request.phase.empty()) {
        return "phase is required";
    }
    if (request.payload_kind.empty()) {
        return "payload_kind is required";
    }
    return "";
}

} // namespace jetsonfabric::runtime::pipeline_parallel
