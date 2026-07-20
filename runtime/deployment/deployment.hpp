#pragma once

#include <optional>
#include <string>

namespace jetsonfabric::runtime::deployment {

enum class ResidentDeploymentState {
    Loading,
    Ready,
    Active,
    Draining,
    Unloading,
    Failed,
};

constexpr bool is_valid_resident_deployment_transition(
    std::optional<ResidentDeploymentState> from,
    std::optional<ResidentDeploymentState> to
) noexcept {
    if (!from.has_value()) {
        return to.has_value() && *to == ResidentDeploymentState::Loading;
    }

    if (!to.has_value()) {
        return *from == ResidentDeploymentState::Unloading;
    }

    switch (*from) {
        case ResidentDeploymentState::Loading:
            return *to == ResidentDeploymentState::Ready ||
                *to == ResidentDeploymentState::Failed;
        case ResidentDeploymentState::Ready:
            return *to == ResidentDeploymentState::Active ||
                *to == ResidentDeploymentState::Unloading ||
                *to == ResidentDeploymentState::Failed;
        case ResidentDeploymentState::Active:
            return *to == ResidentDeploymentState::Draining ||
                *to == ResidentDeploymentState::Failed;
        case ResidentDeploymentState::Draining:
            return *to == ResidentDeploymentState::Unloading ||
                *to == ResidentDeploymentState::Failed;
        case ResidentDeploymentState::Unloading:
            return *to == ResidentDeploymentState::Failed;
        case ResidentDeploymentState::Failed:
            return *to == ResidentDeploymentState::Unloading;
    }

    return false;
}

struct DeploymentIdentity {
    std::string deployment_id;
    std::string model_id;
};

} // namespace jetsonfabric::runtime::deployment
