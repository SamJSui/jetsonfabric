#include "worker/config.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        jetsonfabric::runtime::Config config;
        config.engine = "native";
        config.compute_backend = "cuda";
        config.mode = jetsonfabric::runtime::ExecutionMode::PipelineParallel;
        config.stage_assignment = {
            .stage_index = 1,
            .stage_count = 2,
            .layer_start = 14,
            .layer_end = 28,
        };
        jetsonfabric::runtime::validate_deployment_config(config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native stage config test: " << error.what() << '\n';
        return 1;
    }
}
