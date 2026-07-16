#include "pipeline_parallel/stage_worker.hpp"

#include "inference/stage.hpp"

#include <chrono>
#include <exception>
#include <sstream>
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

int elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
}

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
    int latency_ms
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
    response.completion_tokens = output.completion_tokens;
    response.latency_ms = latency_ms;
    response.message = output.token_text;
    return response;
}

protocol::StageResponse close_response(const protocol::StageRequest& request, int latency_ms) {
    protocol::StageResponse response = base_response(request);
    response.payload_kind = "text";
    response.encoding = "utf-8";
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    response.latency_ms = latency_ms;
    return response;
}

} // namespace

StageWorker::StageWorker(
    std::string node_name,
    StageAssignment assignment,
    const LayerExecutor& layer_executor
)
    : node_name_(std::move(node_name)),
      assignment_(assignment),
      layer_executor_(layer_executor) {}

StageRunResult StageWorker::run(const protocol::StageRequest& request) const {
    const std::string assignment_error = validate_stage_assignment(assignment_);
    if (!assignment_error.empty()) {
        return bad_request("invalid_stage_assignment", assignment_error);
    }

    const std::string request_error = validate_request(request);
    if (!request_error.empty()) {
        return bad_request("invalid_stage_request", request_error);
    }

    inference::StageInput input;
    try {
        input = to_stage_input(request);
    } catch (const std::exception& error) {
        return bad_request("invalid_stage_input", error.what());
    }

    const std::string input_error = inference::validate_stage_input(input);
    if (!input_error.empty()) {
        return bad_request("invalid_stage_input", input_error);
    }

    const auto start = std::chrono::steady_clock::now();
    const inference::ExecutionResult execution = layer_executor_.execute(input);
    const int latency = elapsed_ms(start);
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

    StageRunResult result;
    result.ok = true;
    result.status = "200 OK";
    result.response = to_stage_response(request, execution.output, latency);
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
    result.response = close_response(request, elapsed_ms(start));
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
