#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace jetsonfabric::runtime::deployment {

enum class ResidentDeploymentState {
    Loading,
    Ready,
    Active,
    Draining,
    Unloading,
    Failed,
};

constexpr std::string_view resident_deployment_state_string(
    ResidentDeploymentState state
) noexcept {
    switch (state) {
        case ResidentDeploymentState::Loading:
            return "loading";
        case ResidentDeploymentState::Ready:
            return "ready";
        case ResidentDeploymentState::Active:
            return "active";
        case ResidentDeploymentState::Draining:
            return "draining";
        case ResidentDeploymentState::Unloading:
            return "unloading";
        case ResidentDeploymentState::Failed:
            return "failed";
    }
    return "unknown";
}

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

struct DeploymentStatus {
    bool resident = false;
    bool active = false;
    std::optional<ResidentDeploymentState> state;
    std::optional<DeploymentIdentity> identity;
};

struct UnloadDeploymentResult {
    bool ok = false;
    std::string status;
    std::string error_code;
    std::string error_message;
    std::optional<DeploymentIdentity> identity;
};

} // namespace jetsonfabric::runtime::deployment