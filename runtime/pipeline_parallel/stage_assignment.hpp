#pragma once

#include <string>

namespace jetsonfabric::runtime::pipeline_parallel {

struct StageAssignment {
    int stage_index = 0;
    int stage_count = 1;
    int layer_start = 0;
    int layer_end = 0;

    bool is_first_stage() const;
    bool is_last_stage() const;
};

std::string validate_stage_assignment(const StageAssignment& assignment);

} // namespace jetsonfabric::runtime::pipeline_parallel
