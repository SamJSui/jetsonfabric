#include "pipeline_parallel/stage_assignment.hpp"

#include <sstream>
#include <stdexcept>

namespace jetsonfabric::runtime::pipeline_parallel {

StageRole parse_stage_role(const std::string& value) {
    if (value == "single") {
        return StageRole::Single;
    }
    if (value == "first") {
        return StageRole::First;
    }
    if (value == "middle") {
        return StageRole::Middle;
    }
    if (value == "last") {
        return StageRole::Last;
    }

    throw std::invalid_argument("unknown stage role: " + value);
}

std::string stage_role_string(StageRole role) {
    switch (role) {
    case StageRole::Single:
        return "single";
    case StageRole::First:
        return "first";
    case StageRole::Middle:
        return "middle";
    case StageRole::Last:
        return "last";
    }

    return "single";
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

    if (assignment.layer_end < assignment.layer_start) {
        return "layer_end must be greater than or equal to layer_start";
    }

    if (assignment.stage_count == 1 && assignment.role != StageRole::Single) {
        return "single-stage assignment must use role single";
    }

    if (assignment.stage_count > 1) {
        if (assignment.layer_end <= assignment.layer_start) {
            return "pipeline assignment must have layer_end greater than layer_start";
        }

        if (assignment.stage_index == 0 && assignment.role != StageRole::First) {
            return "stage_index 0 must use role first";
        }

        if (assignment.stage_index == assignment.stage_count - 1 && assignment.role != StageRole::Last) {
            return "last stage must use role last";
        }

        if (assignment.stage_index > 0 &&
            assignment.stage_index < assignment.stage_count - 1 &&
            assignment.role != StageRole::Middle) {
            return "intermediate stages must use role middle";
        }
    }

    return "";
}

} // namespace jetsonfabric::runtime::pipeline_parallel