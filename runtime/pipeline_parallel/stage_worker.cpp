#include "pipeline_parallel/stage_worker.hpp"

#include <sstream>
#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

StageRunResult bad_request(const std::string& code, const std::string& message) {
    StageRunResult result;
    result.ok = false;
    result.status = "400 Bad Request";
    result.error_code = code;
    result.error_message = message;
    return result;
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

StageRunResult StageWorker::run(const protocol::ActivationRequest& request) const {
    const std::string assignment_error = validate_stage_assignment(assignment_);
    if (!assignment_error.empty()) {
        return bad_request("invalid_stage_assignment", assignment_error);
    }

    const std::string request_error = validate_request(request);
    if (!request_error.empty()) {
        return bad_request("invalid_stage_request", request_error);
    }

    return layer_executor_.run_layers(request);
}

std::string StageWorker::validate_request(const protocol::ActivationRequest& request) const {
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
        std::ostringstream msg;
        msg << "request stage_index " << request.stage_index
            << " does not match runtime stage_index " << assignment_.stage_index;
        return msg.str();
    }

    if (request.layer_start != assignment_.layer_start || request.layer_end != assignment_.layer_end) {
        std::ostringstream msg;
        msg << "request layer range [" << request.layer_start << ":" << request.layer_end
            << "] does not match runtime assignment [" << assignment_.layer_start
            << ":" << assignment_.layer_end << "]";
        return msg.str();
    }

    if (request.node_name != node_name_) {
        std::ostringstream msg;
        msg << "request node_name " << request.node_name
            << " does not match runtime node_name " << node_name_;
        return msg.str();
    }

    if (request.role != stage_role_string(assignment_.role)) {
        std::ostringstream msg;
        msg << "request role " << request.role
            << " does not match runtime role " << stage_role_string(assignment_.role);
        return msg.str();
    }

    return "";
}

} // namespace jetsonfabric::runtime::pipeline_parallel