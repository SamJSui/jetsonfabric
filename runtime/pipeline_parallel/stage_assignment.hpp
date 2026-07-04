#pragma once

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

enum class StageRole {
    Single,
    First,
    Middle,
    Last,
};

struct StageAssignment {
    int stage_index = 0;
    int stage_count = 1;
    int layer_start = 0;
    int layer_end = 0;
    StageRole role = StageRole::Single;
};

StageRole parse_stage_role(const std::string& value);
std::string stage_role_string(StageRole role);
std::string validate_stage_assignment(const StageAssignment& assignment);

} // namespace jetsonfabric::runtime::pipeline_parallel