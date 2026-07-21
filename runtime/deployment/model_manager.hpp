#pragma once

#include "deployment/deployment.hpp"
#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

#include <cctype>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::runtime::deployment {

// ModelManager owns one resident deployment slot. A resident identity may exist
// while loading or after a failed load, but inference components are reachable
// only after the deployment is explicitly activated.
class ModelManager {
    struct ResidentDeployment;

public:
    using EngineBuilder = std::function<InferenceEngineParts()>;

    ModelManager() = default;

    explicit ModelManager(const Config& config) {
        if (!config.start_idle) {
            resident_.emplace(
                config.node_name,
                DeploymentIdentity{
                    .deployment_id = config.model,
                    .epoch = 0,
                    .model_id = config.model,
                    .model_sha256 = "",
                },
                config.stage_assignment,
                build_inference_engine_parts(config)
            );
        }
    }

    ModelManager(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        InferenceEngineParts engine_parts
    ) {
        resident_.emplace(
            std::move(node_name),
            std::move(identity),
            assignment,
            std::move(engine_parts)
        );
    }

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;

    bool has_resident_deployment() const noexcept {
        return resident_.has_value();
    }

    bool has_active_deployment() const noexcept {
        return active_deployment() != nullptr;
    }

    const DeploymentIdentity* resident_deployment_identity() const noexcept {
        return resident_.has_value() ? &resident_->identity : nullptr;
    }

    std::optional<ResidentDeploymentState> resident_deployment_state() const noexcept {
        return resident_.has_value()
            ? std::optional<ResidentDeploymentState>{resident_->state}
            : std::nullopt;
    }

    DeploymentStatus deployment_status() const {
        if (!resident_.has_value()) {
            return DeploymentStatus{};
        }
        return DeploymentStatus{
            .resident = true,
            .active = active_deployment() != nullptr,
            .state = resident_->state,
            .identity = resident_->identity,
            .model_residency = resident_->model_residency,
        };
    }

    LoadDeploymentResult load_resident_deployment(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        EngineBuilder build_engine_parts
    ) {
        if (resident_.has_value()) {
            return operation_error(
                "409 Conflict",
                "resident_deployment_exists",
                "runtime already has a resident deployment",
                resident_->identity,
                resident_->state
            );
        }
        if (!build_engine_parts) {
            return operation_error(
                "400 Bad Request",
                "invalid_engine_builder",
                "deployment load requires an engine builder"
            );
        }

        const std::string assignment_error = pipeline_parallel::validate_stage_assignment(assignment);
        if (!assignment_error.empty()) {
            return operation_error(
                "400 Bad Request",
                "invalid_stage_assignment",
                assignment_error
            );
        }

        try {
            resident_.emplace(
                require_identity(std::move(identity)),
                ResidentDeploymentState::Loading
            );
        } catch (const std::invalid_argument& err) {
            return operation_error("400 Bad Request", "invalid_deployment_identity", err.what());
        }

        try {
            InferenceEngineParts engine_parts = build_engine_parts();
            resident_->model_residency = engine_parts.model_residency;
            resident_->execution = std::make_unique<ResidentExecution>(
                std::move(node_name),
                resident_->identity.model_id,
                assignment,
                std::move(engine_parts)
            );
            transition_resident_to(ResidentDeploymentState::Ready);
            return operation_success(
                "200 OK",
                resident_->identity,
                ResidentDeploymentState::Ready
            );
        } catch (const std::exception& err) {
            transition_resident_to(ResidentDeploymentState::Failed);
            return operation_error(
                "500 Internal Server Error",
                "deployment_load_failed",
                err.what(),
                resident_->identity,
                ResidentDeploymentState::Failed
            );
        }
    }

    ActivateDeploymentResult activate_resident_deployment(
        const DeploymentIdentity& expected_identity
    ) {
        try {
            require_managed_identity(expected_identity);
        } catch (const std::invalid_argument& err) {
            return operation_error(
                "400 Bad Request",
                "invalid_deployment_identity",
                err.what()
            );
        }
        if (!resident_.has_value()) {
            return operation_error(
                "409 Conflict",
                "no_resident_deployment",
                "runtime has no resident deployment"
            );
        }
        if (resident_->identity != expected_identity) {
            return operation_error(
                "409 Conflict",
                "deployment_mismatch",
                "resident deployment does not match the expected deployment identity",
                resident_->identity,
                resident_->state
            );
        }
        if (resident_->state == ResidentDeploymentState::Active) {
            return operation_success(
                "200 OK",
                resident_->identity,
                ResidentDeploymentState::Active
            );
        }
        if (resident_->state == ResidentDeploymentState::Loading ||
            resident_->state == ResidentDeploymentState::Draining ||
            resident_->state == ResidentDeploymentState::Unloading) {
            return operation_error(
                "503 Service Unavailable",
                "deployment_transitioning",
                "resident deployment is transitioning",
                resident_->identity,
                resident_->state
            );
        }
        if (resident_->state == ResidentDeploymentState::Failed) {
            return operation_error(
                "409 Conflict",
                "deployment_failed",
                "failed deployment must be unloaded before another load",
                resident_->identity,
                resident_->state
            );
        }
        if (resident_->state != ResidentDeploymentState::Ready ||
            !resident_->execution) {
            return operation_error(
                "500 Internal Server Error",
                "invalid_deployment_state",
                "ready deployment is missing execution components",
                resident_->identity,
                resident_->state
            );
        }

        transition_resident_to(ResidentDeploymentState::Active);
        return operation_success(
            "200 OK",
            resident_->identity,
            ResidentDeploymentState::Active
        );
    }

    UnloadDeploymentResult unload_resident_deployment(
        const DeploymentIdentity& expected_identity
    ) {
        try {
            require_managed_identity(expected_identity);
        } catch (const std::invalid_argument& err) {
            return operation_error(
                "400 Bad Request",
                "invalid_deployment_identity",
                err.what()
            );
        }
        if (!resident_.has_value()) {
            return operation_error(
                "409 Conflict",
                "no_resident_deployment",
                "runtime has no resident deployment"
            );
        }
        if (resident_->identity != expected_identity) {
            return operation_error(
                "409 Conflict",
                "deployment_mismatch",
                "resident deployment does not match the expected deployment identity",
                resident_->identity,
                resident_->state
            );
        }
        if (resident_->state == ResidentDeploymentState::Loading ||
            resident_->state == ResidentDeploymentState::Unloading) {
            return operation_error(
                "503 Service Unavailable",
                "deployment_transitioning",
                "resident deployment is already transitioning",
                resident_->identity,
                resident_->state
            );
        }

        if (resident_->state == ResidentDeploymentState::Active) {
            transition_resident_to(ResidentDeploymentState::Draining);
        }
        transition_resident_to(ResidentDeploymentState::Unloading);

        DeploymentIdentity unloaded_identity = resident_->identity;
        if (!is_valid_resident_deployment_transition(
                ResidentDeploymentState::Unloading,
                std::nullopt
            )) {
            throw std::logic_error("unloading deployment cannot transition to idle");
        }
        resident_.reset();

        return operation_success("200 OK", std::move(unloaded_identity), std::nullopt);
    }

    const DeploymentIdentity* active_deployment_identity() const noexcept {
        const ResidentDeployment* deployment = active_deployment();
        return deployment != nullptr ? &deployment->identity : nullptr;
    }

    const std::string& active_deployment_id() const noexcept {
        static const std::string empty_deployment_id;
        const DeploymentIdentity* identity = active_deployment_identity();
        return identity != nullptr ? identity->deployment_id : empty_deployment_id;
    }

    const std::string& active_model_id() const noexcept {
        static const std::string empty_model_id;
        const DeploymentIdentity* identity = active_deployment_identity();
        return identity != nullptr ? identity->model_id : empty_model_id;
    }

    pipeline_parallel::StageRunResult run_stage(const protocol::StageRequest& request) const {
        const ResidentDeployment* deployment = active_deployment();
        if (deployment == nullptr) {
            return no_active_deployment();
        }
        return deployment->execution->stage_worker.run(request);
    }

    pipeline_parallel::StageRunResult close_session(const protocol::StageRequest& request) const {
        const ResidentDeployment* deployment = active_deployment();
        if (deployment == nullptr) {
            return no_active_deployment();
        }
        return deployment->execution->stage_worker.close_session(request);
    }

private:
    struct ResidentExecution {
        ResidentExecution(
            std::string node_name,
            std::string model_id,
            pipeline_parallel::StageAssignment assignment,
            InferenceEngineParts loaded_engine_parts
        )
            : engine_parts(std::move(loaded_engine_parts)),
              stage_worker(
                  std::move(node_name),
                  std::move(model_id),
                  assignment,
                  ModelManager::require_layer_executor(engine_parts)
              ) {}

        ResidentExecution(const ResidentExecution&) = delete;
        ResidentExecution& operator=(const ResidentExecution&) = delete;
        ResidentExecution(ResidentExecution&&) = delete;
        ResidentExecution& operator=(ResidentExecution&&) = delete;

        InferenceEngineParts engine_parts;
        pipeline_parallel::StageWorker stage_worker;
    };

    struct ResidentDeployment {
        ResidentDeployment(
            DeploymentIdentity deployment_identity,
            ResidentDeploymentState deployment_state
        )
            : identity(ModelManager::require_identity(std::move(deployment_identity))),
              state(deployment_state) {}

        ResidentDeployment(
            std::string node_name,
            DeploymentIdentity deployment_identity,
            pipeline_parallel::StageAssignment assignment,
            InferenceEngineParts loaded_engine_parts
        )
            : identity(ModelManager::require_identity(std::move(deployment_identity))),
              state(ResidentDeploymentState::Active),
              model_residency(loaded_engine_parts.model_residency),
              execution(std::make_unique<ResidentExecution>(
                  std::move(node_name),
                  identity.model_id,
                  assignment,
                  std::move(loaded_engine_parts)
              )) {}

        ResidentDeployment(const ResidentDeployment&) = delete;
        ResidentDeployment& operator=(const ResidentDeployment&) = delete;
        ResidentDeployment(ResidentDeployment&&) = delete;
        ResidentDeployment& operator=(ResidentDeployment&&) = delete;

        DeploymentIdentity identity;
        ResidentDeploymentState state;
        std::optional<ModelResidency> model_residency;
        std::unique_ptr<ResidentExecution> execution;
    };

    static pipeline_parallel::StageRunResult no_active_deployment() {
        pipeline_parallel::StageRunResult result;
        result.ok = false;
        result.status = "503 Service Unavailable";
        result.error_code = "no_active_deployment";
        result.error_message = "runtime has no active deployment";
        return result;
    }

    static DeploymentOperationResult operation_success(
        std::string status,
        DeploymentIdentity identity,
        std::optional<ResidentDeploymentState> state
    ) {
        return DeploymentOperationResult{
            .ok = true,
            .status = std::move(status),
            .identity = std::move(identity),
            .state = state,
        };
    }

    static DeploymentOperationResult operation_error(
        std::string status,
        std::string code,
        std::string message,
        std::optional<DeploymentIdentity> identity = std::nullopt,
        std::optional<ResidentDeploymentState> state = std::nullopt
    ) {
        return DeploymentOperationResult{
            .ok = false,
            .status = std::move(status),
            .error_code = std::move(code),
            .error_message = std::move(message),
            .identity = std::move(identity),
            .state = state,
        };
    }

    static DeploymentIdentity require_identity(DeploymentIdentity identity) {
        if (identity.deployment_id.empty()) {
            throw std::invalid_argument("deployment_id is required");
        }
        if (identity.model_id.empty()) {
            throw std::invalid_argument("model_id is required");
        }
        return identity;
    }

    static const DeploymentIdentity& require_managed_identity(const DeploymentIdentity& identity) {
        require_identity(identity);
        if (identity.epoch == 0) {
            throw std::invalid_argument("deployment epoch must be positive");
        }
        if (identity.model_sha256.size() != 64) {
            throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
        }
        for (const unsigned char character : identity.model_sha256) {
            if (!std::isxdigit(character)) {
                throw std::invalid_argument("model_sha256 must be a 64-character hexadecimal digest");
            }
        }
        return identity;
    }

    static const pipeline_parallel::LayerExecutor& require_layer_executor(
        const InferenceEngineParts& engine_parts
    ) {
        if (!engine_parts.layer_executor) {
            throw std::invalid_argument("model manager requires a layer executor");
        }
        return *engine_parts.layer_executor;
    }

    void transition_resident_to(ResidentDeploymentState next) {
        if (!resident_.has_value() ||
            !is_valid_resident_deployment_transition(resident_->state, next)) {
            throw std::logic_error("invalid resident deployment transition");
        }
        resident_->state = next;
    }

    const ResidentDeployment* active_deployment() const noexcept {
        if (!resident_.has_value() ||
            resident_->state != ResidentDeploymentState::Active ||
            !resident_->execution) {
            return nullptr;
        }
        return &*resident_;
    }

    std::optional<ResidentDeployment> resident_;
};

} // namespace jetsonfabric::runtime::deployment
