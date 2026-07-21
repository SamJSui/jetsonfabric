#pragma once

#include "deployment/deployment.hpp"
#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace jetsonfabric::runtime::deployment {

// ModelManager owns resident deployments on one runtime. Managed requests carry
// an immutable deployment identity, allowing a replacement epoch to be prepared
// while the previous epoch remains executable for already-admitted sessions.
class ModelManager {
    struct ResidentDeployment;
    using DeploymentKey = std::pair<std::string, std::uint64_t>;

public:
    using EngineBuilder = std::function<InferenceEngineParts()>;

    ModelManager() = default;

    explicit ModelManager(const Config& config) {
        if (!config.start_idle) {
            insert_active(
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
        insert_active(
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
        return !deployments_.empty();
    }

    std::size_t resident_deployment_count() const noexcept {
        return deployments_.size();
    }

    bool has_active_deployment() const noexcept {
        for (const auto& [key, deployment] : deployments_) {
            (void) key;
            if (is_executable(*deployment)) {
                return true;
            }
        }
        return false;
    }

    const DeploymentIdentity* resident_deployment_identity() const noexcept {
        const ResidentDeployment* deployment = preferred_deployment();
        return deployment != nullptr ? &deployment->identity : nullptr;
    }

    std::optional<ResidentDeploymentState> resident_deployment_state() const noexcept {
        const ResidentDeployment* deployment = preferred_deployment();
        return deployment != nullptr
            ? std::optional<ResidentDeploymentState>{deployment->state}
            : std::nullopt;
    }

    DeploymentStatus deployment_status() const {
        return status_for(preferred_deployment());
    }

    DeploymentStatus deployment_status(const DeploymentIdentity& identity) const {
        return status_for(find_exact(identity));
    }

    LoadDeploymentResult load_resident_deployment(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        EngineBuilder build_engine_parts
    ) {
        if (!build_engine_parts) {
            return operation_error(
                "400 Bad Request",
                "invalid_engine_builder",
                "deployment load requires an engine builder"
            );
        }
        const std::string assignment_error =
            pipeline_parallel::validate_stage_assignment(assignment);
        if (!assignment_error.empty()) {
            return operation_error(
                "400 Bad Request",
                "invalid_stage_assignment",
                assignment_error
            );
        }

        try {
            require_managed_identity(identity);
        } catch (const std::invalid_argument& error) {
            return operation_error(
                "400 Bad Request",
                "invalid_deployment_identity",
                error.what()
            );
        }

        const DeploymentKey key = key_for(identity);
        if (const ResidentDeployment* existing = find_key(key); existing != nullptr) {
            return operation_error(
                "409 Conflict",
                "resident_deployment_exists",
                "runtime already has this resident deployment identity",
                existing->identity,
                existing->state
            );
        }

        auto deployment = std::make_unique<ResidentDeployment>(
            std::move(identity),
            ResidentDeploymentState::Loading
        );
        ResidentDeployment* resident = deployment.get();
        deployments_.emplace(key, std::move(deployment));
        if (!preferred_.has_value()) {
            preferred_ = key;
        }

        try {
            InferenceEngineParts engine_parts = build_engine_parts();
            resident->model_residency = engine_parts.model_residency;
            resident->execution = std::make_unique<ResidentExecution>(
                std::move(node_name),
                resident->identity.model_id,
                assignment,
                std::move(engine_parts)
            );
            transition(*resident, ResidentDeploymentState::Ready);
            return operation_success(
                "200 OK",
                resident->identity,
                ResidentDeploymentState::Ready
            );
        } catch (const std::exception& error) {
            transition(*resident, ResidentDeploymentState::Failed);
            return operation_error(
                "500 Internal Server Error",
                "deployment_load_failed",
                error.what(),
                resident->identity,
                ResidentDeploymentState::Failed
            );
        }
    }

    ActivateDeploymentResult activate_resident_deployment(
        const DeploymentIdentity& expected_identity
    ) {
        if (DeploymentOperationResult invalid = validate_operation_identity(expected_identity);
            !invalid.ok) {
            return invalid;
        }
        ResidentDeployment* resident = find_exact(expected_identity);
        if (resident == nullptr) {
            return deployment_not_found(expected_identity);
        }
        if (resident->state == ResidentDeploymentState::Active) {
            preferred_ = key_for(resident->identity);
            return operation_success("200 OK", resident->identity, resident->state);
        }
        if (resident->state == ResidentDeploymentState::Loading ||
            resident->state == ResidentDeploymentState::Draining ||
            resident->state == ResidentDeploymentState::Unloading) {
            return transitioning_error(*resident);
        }
        if (resident->state == ResidentDeploymentState::Failed) {
            return operation_error(
                "409 Conflict",
                "deployment_failed",
                "failed deployment must be unloaded before activation",
                resident->identity,
                resident->state
            );
        }
        if (resident->state != ResidentDeploymentState::Ready ||
            !resident->execution) {
            return invalid_state_error(*resident);
        }

        transition(*resident, ResidentDeploymentState::Active);
        preferred_ = key_for(resident->identity);
        return operation_success("200 OK", resident->identity, resident->state);
    }

    DrainDeploymentResult drain_resident_deployment(
        const DeploymentIdentity& expected_identity
    ) {
        if (DeploymentOperationResult invalid = validate_operation_identity(expected_identity);
            !invalid.ok) {
            return invalid;
        }
        ResidentDeployment* resident = find_exact(expected_identity);
        if (resident == nullptr) {
            return operation_success("200 OK", expected_identity, std::nullopt);
        }
        if (resident->state == ResidentDeploymentState::Loading ||
            resident->state == ResidentDeploymentState::Unloading) {
            return transitioning_error(*resident);
        }
        if (resident->state == ResidentDeploymentState::Active) {
            transition(*resident, ResidentDeploymentState::Draining);
        }
        return operation_success("200 OK", resident->identity, resident->state);
    }

    UnloadDeploymentResult unload_resident_deployment(
        const DeploymentIdentity& expected_identity
    ) {
        if (DeploymentOperationResult invalid = validate_operation_identity(expected_identity);
            !invalid.ok) {
            return invalid;
        }
        ResidentDeployment* resident = find_exact(expected_identity);
        if (resident == nullptr) {
            return operation_success("200 OK", expected_identity, std::nullopt);
        }
        if (resident->state == ResidentDeploymentState::Loading ||
            resident->state == ResidentDeploymentState::Unloading) {
            return transitioning_error(*resident);
        }
        if (resident->state == ResidentDeploymentState::Active) {
            return operation_error(
                "409 Conflict",
                "deployment_not_draining",
                "active deployment must enter draining state before unload",
                resident->identity,
                resident->state
            );
        }

        const DeploymentKey key = key_for(resident->identity);
        const DeploymentIdentity unloaded_identity = resident->identity;
        transition(*resident, ResidentDeploymentState::Unloading);
        deployments_.erase(key);
        select_preferred_after_erase();
        return operation_success("200 OK", unloaded_identity, std::nullopt);
    }

    const DeploymentIdentity* active_deployment_identity() const noexcept {
        const ResidentDeployment* deployment = preferred_deployment();
        return deployment != nullptr && is_executable(*deployment)
            ? &deployment->identity
            : nullptr;
    }

    const DeploymentIdentity* executable_deployment_identity(
        const DeploymentIdentity& expected_identity
    ) const noexcept {
        const ResidentDeployment* deployment = find_exact(expected_identity);
        return deployment != nullptr && is_executable(*deployment)
            ? &deployment->identity
            : nullptr;
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

    pipeline_parallel::StageRunResult run_stage(
        const protocol::StageRequest& request
    ) const {
        const ResidentDeployment* deployment = deployment_for(request);
        if (deployment == nullptr) {
            return unavailable_for(request);
        }
        if (pipeline_parallel::StageRunResult mismatch =
                validate_stage_deployment(deployment->identity, request);
            !mismatch.ok) {
            return mismatch;
        }
        return deployment->execution->stage_worker.run(request);
    }

    pipeline_parallel::StageRunResult close_session(
        const protocol::StageRequest& request
    ) const {
        const ResidentDeployment* deployment = deployment_for(request);
        if (deployment == nullptr) {
            return unavailable_for(request);
        }
        if (pipeline_parallel::StageRunResult mismatch =
                validate_stage_deployment(deployment->identity, request);
            !mismatch.ok) {
            return mismatch;
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

        DeploymentIdentity identity;
        ResidentDeploymentState state;
        std::optional<ModelResidency> model_residency;
        std::unique_ptr<ResidentExecution> execution;
    };

    void insert_active(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        InferenceEngineParts engine_parts
    ) {
        const DeploymentKey key = key_for(identity);
        deployments_.emplace(
            key,
            std::make_unique<ResidentDeployment>(
                std::move(node_name),
                std::move(identity),
                assignment,
                std::move(engine_parts)
            )
        );
        preferred_ = key;
    }

    static DeploymentKey key_for(const DeploymentIdentity& identity) {
        return {identity.deployment_id, identity.epoch};
    }

    ResidentDeployment* find_key(const DeploymentKey& key) noexcept {
        const auto found = deployments_.find(key);
        return found != deployments_.end() ? found->second.get() : nullptr;
    }

    const ResidentDeployment* find_key(const DeploymentKey& key) const noexcept {
        const auto found = deployments_.find(key);
        return found != deployments_.end() ? found->second.get() : nullptr;
    }

    ResidentDeployment* find_exact(const DeploymentIdentity& identity) noexcept {
        ResidentDeployment* deployment = find_key(key_for(identity));
        return deployment != nullptr && deployment->identity == identity
            ? deployment
            : nullptr;
    }

    const ResidentDeployment* find_exact(
        const DeploymentIdentity& identity
    ) const noexcept {
        const ResidentDeployment* deployment = find_key(key_for(identity));
        return deployment != nullptr && deployment->identity == identity
            ? deployment
            : nullptr;
    }

    const ResidentDeployment* preferred_deployment() const noexcept {
        return preferred_.has_value() ? find_key(*preferred_) : nullptr;
    }

    const ResidentDeployment* deployment_for(
        const protocol::StageRequest& request
    ) const noexcept {
        const bool managed = !request.deployment_id.empty() ||
            request.deployment_epoch != 0 ||
            !request.model_sha256.empty();
        if (!managed) {
            const ResidentDeployment* deployment = preferred_deployment();
            return deployment != nullptr && is_executable(*deployment)
                ? deployment
                : nullptr;
        }
        const DeploymentIdentity expected{
            .deployment_id = request.deployment_id,
            .epoch = request.deployment_epoch,
            .model_id = request.model_id,
            .model_sha256 = request.model_sha256,
        };
        const ResidentDeployment* deployment = find_exact(expected);
        return deployment != nullptr && is_executable(*deployment)
            ? deployment
            : nullptr;
    }

    static bool is_executable(const ResidentDeployment& deployment) noexcept {
        return deployment.execution &&
            (deployment.state == ResidentDeploymentState::Active ||
             deployment.state == ResidentDeploymentState::Draining);
    }

    static DeploymentStatus status_for(const ResidentDeployment* deployment) {
        if (deployment == nullptr) {
            return DeploymentStatus{};
        }
        return DeploymentStatus{
            .resident = true,
            .active = is_executable(*deployment),
            .state = deployment->state,
            .identity = deployment->identity,
            .model_residency = deployment->model_residency,
        };
    }

    void select_preferred_after_erase() noexcept {
        preferred_.reset();
        for (auto iterator = deployments_.rbegin(); iterator != deployments_.rend(); ++iterator) {
            if (is_executable(*iterator->second)) {
                preferred_ = iterator->first;
                return;
            }
        }
        if (!deployments_.empty()) {
            preferred_ = deployments_.rbegin()->first;
        }
    }

    static DeploymentOperationResult validate_operation_identity(
        const DeploymentIdentity& identity
    ) {
        try {
            require_managed_identity(identity);
            DeploymentOperationResult result;
            result.ok = true;
            return result;
        } catch (const std::invalid_argument& error) {
            return operation_error(
                "400 Bad Request",
                "invalid_deployment_identity",
                error.what()
            );
        }
    }

    static DeploymentOperationResult deployment_not_found(
        const DeploymentIdentity& identity
    ) {
        return operation_error(
            "409 Conflict",
            "deployment_not_found",
            "runtime has no resident deployment matching the expected identity",
            identity
        );
    }

    static DeploymentOperationResult transitioning_error(
        const ResidentDeployment& deployment
    ) {
        return operation_error(
            "503 Service Unavailable",
            "deployment_transitioning",
            "resident deployment is transitioning",
            deployment.identity,
            deployment.state
        );
    }

    static DeploymentOperationResult invalid_state_error(
        const ResidentDeployment& deployment
    ) {
        return operation_error(
            "500 Internal Server Error",
            "invalid_deployment_state",
            "ready deployment is missing execution components",
            deployment.identity,
            deployment.state
        );
    }

    static pipeline_parallel::StageRunResult no_active_deployment() {
        pipeline_parallel::StageRunResult result;
        result.ok = false;
        result.status = "503 Service Unavailable";
        result.error_code = "no_active_deployment";
        result.error_message = "runtime has no executable deployment for the request identity";
        return result;
    }

    pipeline_parallel::StageRunResult unavailable_for(
        const protocol::StageRequest& request
    ) const {
        const bool managed = !request.deployment_id.empty() ||
            request.deployment_epoch != 0 ||
            !request.model_sha256.empty();
        if (!managed || !has_active_deployment()) {
            return no_active_deployment();
        }
        pipeline_parallel::StageRunResult result;
        result.ok = false;
        result.status = "409 Conflict";
        result.error_code = "deployment_mismatch";
        result.error_message =
            "stage request identity does not match a resident executable epoch";
        return result;
    }

    static pipeline_parallel::StageRunResult stage_success() {
        pipeline_parallel::StageRunResult result;
        result.ok = true;
        return result;
    }

    static pipeline_parallel::StageRunResult validate_stage_deployment(
        const DeploymentIdentity& active,
        const protocol::StageRequest& request
    ) {
        const bool request_has_identity = !request.deployment_id.empty() ||
            request.deployment_epoch != 0 ||
            !request.model_sha256.empty();
        if (active.epoch == 0 && !request_has_identity) {
            return stage_success();
        }
        if (request.deployment_id == active.deployment_id &&
            request.deployment_epoch == active.epoch &&
            request.model_id == active.model_id &&
            request.model_sha256 == active.model_sha256) {
            return stage_success();
        }
        pipeline_parallel::StageRunResult result;
        result.ok = false;
        result.status = "409 Conflict";
        result.error_code = "deployment_mismatch";
        result.error_message =
            "stage request deployment identity does not match the selected runtime epoch";
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
            .error_code = "",
            .error_message = "",
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

    static const DeploymentIdentity& require_managed_identity(
        const DeploymentIdentity& identity
    ) {
        require_identity(identity);
        if (identity.epoch == 0) {
            throw std::invalid_argument("deployment epoch must be positive");
        }
        if (identity.model_sha256.size() != 64) {
            throw std::invalid_argument(
                "model_sha256 must be a 64-character hexadecimal digest"
            );
        }
        for (const unsigned char character : identity.model_sha256) {
            if (!std::isxdigit(character)) {
                throw std::invalid_argument(
                    "model_sha256 must be a 64-character hexadecimal digest"
                );
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

    static void transition(
        ResidentDeployment& deployment,
        ResidentDeploymentState next
    ) {
        if (!is_valid_resident_deployment_transition(deployment.state, next)) {
            throw std::logic_error("invalid resident deployment transition");
        }
        deployment.state = next;
    }

    std::map<DeploymentKey, std::unique_ptr<ResidentDeployment>> deployments_;
    std::optional<DeploymentKey> preferred_;
};

} // namespace jetsonfabric::runtime::deployment
