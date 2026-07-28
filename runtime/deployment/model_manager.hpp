#pragma once

#include "deployment/deployment.hpp"
#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_assignment.hpp"
#include "pipeline_parallel/stage_result.hpp"
#include "protocol/stage.hpp"
#include "worker/config.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace jetsonfabric::runtime::deployment {

// Owns resident model epochs and routes stage work to the selected epoch.
// The implementation is private so callers see lifecycle operations, not
// deployment storage and transition mechanics.
class ModelManager {
public:
    using EngineBuilder = std::function<InferenceEngineParts()>;

    ModelManager();
    ModelManager(const Config& config, const InferenceEngineFactory& engine_factory);
    ModelManager(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        InferenceEngineParts engine_parts
    );
    ~ModelManager();

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;

    bool has_resident_deployment() const noexcept;
    std::size_t resident_deployment_count() const noexcept;
    bool has_active_deployment() const noexcept;

    const DeploymentIdentity* resident_deployment_identity() const noexcept;
    std::optional<ResidentDeploymentState> resident_deployment_state() const noexcept;
    DeploymentStatus deployment_status() const;
    DeploymentStatus deployment_status(const DeploymentIdentity& identity) const;

    LoadDeploymentResult load_resident_deployment(
        std::string node_name,
        DeploymentIdentity identity,
        pipeline_parallel::StageAssignment assignment,
        EngineBuilder build_engine_parts
    );
    ActivateDeploymentResult activate_resident_deployment(
        const DeploymentIdentity& expected_identity
    );
    DrainDeploymentResult drain_resident_deployment(
        const DeploymentIdentity& expected_identity
    );
    UnloadDeploymentResult unload_resident_deployment(
        const DeploymentIdentity& expected_identity
    );

    const DeploymentIdentity* active_deployment_identity() const noexcept;
    const DeploymentIdentity* executable_deployment_identity(
        const DeploymentIdentity& expected_identity
    ) const noexcept;
    const std::string& active_deployment_id() const noexcept;
    const std::string& active_model_id() const noexcept;

    pipeline_parallel::StageRunResult run_stage(
        const protocol::StageRequest& request
    ) const;
    pipeline_parallel::StageRunResult close_session(
        const protocol::StageRequest& request
    ) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::deployment
