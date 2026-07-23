#include "pipeline_parallel/stage_assignment.hpp"

#include <sstream>

namespace jetsonfabric::runtime::pipeline_parallel {

bool StageAssignment::is_first_stage() const {
    return stage_count > 0 && stage_index == 0;
}

bool StageAssignment::is_last_stage() const {
    return stage_count > 0 && stage_index == stage_count - 1;
}

std::string validate_stage_assignment(const StageAssignment& assignment) {
    if (assignment.stage_count < 1) {
        return "stage_count must be at least 1";
    }
    if (assignment.stage_index < 0 || assignment.stage_index >= assignment.stage_count) {
        std::ostringstream msg;
        msg << "stage_index " << assignment.stage_index
            << " is outside stage_count " << assignment.stage_count;
        return msg.str();
    }
    if (assignment.layer_end <= assignment.layer_start) {
        return "layer_end must be greater than layer_start";
    }
    return "";
}

} // namespace jetsonfabric::runtime::pipeline_parallel
