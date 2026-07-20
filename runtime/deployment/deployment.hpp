#pragma once

#include <string>

namespace jetsonfabric::runtime::deployment {

enum class DeploymentState {
    Prepared,
    Active,
};

struct DeploymentIdentity {
    std::string deployment_id;
    std::string model_id;
};

} // namespace jetsonfabric::runtime::deployment
