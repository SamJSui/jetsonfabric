#include "deployment/model_manager.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace runtime = jetsonfabric::runtime;

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

class RecordingExecutor final : public runtime::pipeline_parallel::LayerExecutor {
public:
    explicit RecordingExecutor(int* destruction_count)
        : destruction_count_(destruction_count) {}

    ~RecordingExecutor() override {
        ++*destruction_count_;
    }

    runtime::inference::ExecutionResult execute(
        const runtime::inference::StageInput& input
    ) const override {
        ++execute_calls;
        last_model_id = input.model_id;
        return runtime::inference::ExecutionResult::success(runtime::inference::StageOutput{
            .payload = runtime::inference::Payload{
                .kind = runtime::inference::PayloadKind::SampledToken,
                .tensor = runtime::inference::TensorDescriptor{
                    .dtype = "u32",
                    .shape = {1},
                    .byte_order = "little",
                    .layout = "row_major",
                },
                .bytes = {42, 0, 0, 0},
            },
            .completion_tokens = 1,
        });
    }

    void close_session(const std::string&) const override {}

    mutable int execute_calls = 0;
    mutable std::string last_model_id;

private:
    int* destruction_count_;
};

runtime::pipeline_parallel::StageAssignment assignment() {
    return runtime::pipeline_parallel::StageAssignment{
        .stage_index = 0,
        .stage_count = 1,
        .layer_start = 0,
        .layer_end = 4,
    };
}

runtime::deployment::DeploymentIdentity identity(
    std::string deployment_id = "deployment-a",
    std::uint64_t epoch = 1,
    std::string model_id = "model-a",
    char digest_character = 'a'
) {
    return runtime::deployment::DeploymentIdentity{
        .deployment_id = std::move(deployment_id),
        .epoch = epoch,
        .model_id = std::move(model_id),
        .model_sha256 = std::string(64, digest_character),
    };
}

runtime::deployment::ModelResidency residency() {
    return runtime::deployment::ModelResidency{
        .layer_start = 0,
        .layer_end = 4,
        .layer_count = 8,
        .resident_weight_bytes = 400,
        .total_weight_bytes = 800,
        .resident_tensor_count = 20,
    };
}

runtime::protocol::StageRequest request(
    runtime::deployment::DeploymentIdentity deployment = identity()
) {
    return runtime::protocol::StageRequest{
        .session_id = "session-lifecycle",
        .request_id = "request-lifecycle",
        .model_id = deployment.model_id,
        .deployment_id = deployment.deployment_id,
        .deployment_epoch = deployment.epoch,
        .model_sha256 = deployment.model_sha256,
        .phase = "prefill",
        .stage_index = 0,
        .stage_count = 1,
        .node_name = "node-a",
        .layer_start = 0,
        .layer_end = 4,
        .payload_kind = "text",
        .encoding = "utf-8",
        .payload = {'h', 'i'},
        .max_tokens = 1,
    };
}

void test_prepare_activate_drain_handoff() {
    runtime::deployment::ModelManager manager;
    int build_calls = 0;
    int destruction_count = 0;
    RecordingExecutor* recording_a = nullptr;
    RecordingExecutor* recording_b = nullptr;

    const runtime::deployment::LoadDeploymentResult loaded =
        manager.load_resident_deployment(
            "node-a",
            identity(),
            assignment(),
            [&]() {
                ++build_calls;
                auto executor = std::make_unique<RecordingExecutor>(&destruction_count);
                recording_a = executor.get();
                return runtime::InferenceEngineParts{
                    .layer_executor = std::move(executor),
                    .model_residency = residency(),
                };
            }
        );

    expect(loaded.ok, "idle runtime failed to load a deployment");
    expect(build_calls == 1, "deployment engine was not built exactly once");
    expect(manager.has_resident_deployment(), "loaded deployment was not resident");
    expect(!manager.has_active_deployment(), "loaded deployment activated before barrier");
    expect(
        manager.resident_deployment_state() == runtime::deployment::ResidentDeploymentState::Ready,
        "loaded deployment did not become ready"
    );
    const runtime::deployment::DeploymentStatus ready_status = manager.deployment_status();
    expect(ready_status.model_residency.has_value(), "ready deployment omitted model residency");
    expect(ready_status.model_residency->partitioned(), "ready deployment lost partition identity");
    expect(
        ready_status.model_residency->resident_weight_bytes == 400,
        "ready deployment reported the wrong resident bytes"
    );

    const runtime::pipeline_parallel::StageRunResult before_activation =
        manager.run_stage(request());
    expect(!before_activation.ok, "ready deployment accepted inference before activation");
    expect(
        before_activation.error_code == "no_active_deployment",
        "pre-activation inference used the wrong error"
    );

    int rejected_builder_calls = 0;
    const runtime::deployment::LoadDeploymentResult duplicate =
        manager.load_resident_deployment(
            "node-a",
            identity(),
            assignment(),
            [&]() {
                ++rejected_builder_calls;
                return runtime::InferenceEngineParts{};
            }
        );
    expect(!duplicate.ok, "runtime accepted a duplicate deployment identity");
    expect(duplicate.error_code == "resident_deployment_exists", "duplicate load used the wrong error");
    expect(rejected_builder_calls == 0, "duplicate identity rebuilt its engine");

    const runtime::deployment::ActivateDeploymentResult stale =
        manager.activate_resident_deployment(identity("deployment-missing", 2));
    expect(!stale.ok, "stale activation command succeeded");
    expect(stale.error_code == "deployment_not_found", "stale activation used the wrong error");

    const runtime::deployment::ActivateDeploymentResult stale_epoch =
        manager.activate_resident_deployment(identity("deployment-a", 2));
    expect(!stale_epoch.ok, "activation accepted a stale deployment epoch");
    expect(stale_epoch.error_code == "deployment_not_found", "stale epoch used the wrong error");

    const runtime::deployment::ActivateDeploymentResult wrong_artifact =
        manager.activate_resident_deployment(identity("deployment-a", 1, "model-a", 'b'));
    expect(!wrong_artifact.ok, "activation accepted a different model artifact digest");
    expect(wrong_artifact.error_code == "deployment_not_found", "artifact mismatch used the wrong error");

    const runtime::deployment::ActivateDeploymentResult activated =
        manager.activate_resident_deployment(identity());
    expect(activated.ok, "ready deployment failed to activate");
    expect(manager.has_active_deployment(), "activated deployment was not active");
    expect(
        manager.deployment_status().model_residency.has_value(),
        "activation lost model residency accounting"
    );

    const runtime::deployment::ActivateDeploymentResult repeated_activation =
        manager.activate_resident_deployment(identity());
    expect(repeated_activation.ok, "activation retry was not idempotent");

    const runtime::pipeline_parallel::StageRunResult inference_a = manager.run_stage(request());
    expect(inference_a.ok, "active deployment failed inference");
    expect(recording_a != nullptr && recording_a->execute_calls == 1, "active executor was not called once");
    expect(recording_a->last_model_id == "model-a", "active executor received the wrong model");

    const auto identity_b = identity("deployment-b", 2, "model-b", 'b');
    const runtime::deployment::LoadDeploymentResult prepared_b =
        manager.load_resident_deployment(
            "node-a",
            identity_b,
            assignment(),
            [&]() {
                ++build_calls;
                auto executor = std::make_unique<RecordingExecutor>(&destruction_count);
                recording_b = executor.get();
                return runtime::InferenceEngineParts{
                    .layer_executor = std::move(executor),
                    .model_residency = residency(),
                };
            }
        );
    expect(prepared_b.ok, "replacement epoch could not be prepared beside the active epoch");
    expect(build_calls == 2, "replacement engine was not built exactly once");
    expect(manager.resident_deployment_count() == 2, "runtime did not retain both epochs");
    expect(
        manager.deployment_status(identity()).state == runtime::deployment::ResidentDeploymentState::Active,
        "preparing the replacement changed the old epoch"
    );
    expect(
        manager.deployment_status(identity_b).state == runtime::deployment::ResidentDeploymentState::Ready,
        "replacement epoch did not stop at the ready barrier"
    );

    const runtime::deployment::ActivateDeploymentResult activated_b =
        manager.activate_resident_deployment(identity_b);
    expect(activated_b.ok, "replacement epoch failed to activate");
    expect(manager.active_deployment_id() == "deployment-b", "replacement was not preferred after activation");

    const runtime::deployment::DrainDeploymentResult drained_a =
        manager.drain_resident_deployment(identity());
    expect(drained_a.ok, "old epoch failed to enter draining state");
    const runtime::deployment::DeploymentStatus draining_status =
        manager.deployment_status(identity());
    expect(
        draining_status.state == runtime::deployment::ResidentDeploymentState::Draining &&
            draining_status.active,
        "draining epoch was not kept executable"
    );
    expect(manager.run_stage(request()).ok, "pinned old-epoch work failed during drain");
    expect(manager.run_stage(request(identity_b)).ok, "new-epoch work failed after publication");
    expect(recording_a->execute_calls == 2, "old epoch did not receive its pinned request");
    expect(recording_b != nullptr && recording_b->execute_calls == 1, "new epoch did not receive its request");

    const runtime::deployment::UnloadDeploymentResult unloaded_a =
        manager.unload_resident_deployment(identity());
    expect(unloaded_a.ok, "drained old epoch failed to unload");
    expect(destruction_count == 1, "old epoch resources were not released exactly once");
    expect(manager.resident_deployment_count() == 1, "old epoch unload removed the replacement");
    expect(manager.active_deployment_id() == "deployment-b", "replacement stopped serving after old unload");

    expect(manager.drain_resident_deployment(identity_b).ok, "replacement failed to drain");
    expect(manager.unload_resident_deployment(identity_b).ok, "replacement failed to unload");
    expect(destruction_count == 2, "final unload did not release both execution components");
    expect(!manager.has_resident_deployment(), "final unload did not return the runtime to idle");
    expect(!manager.has_active_deployment(), "final unload left an executable deployment");
}

void test_failed_load_is_visible_and_recoverable_by_unload() {
    runtime::deployment::ModelManager manager;

    const runtime::deployment::LoadDeploymentResult failed =
        manager.load_resident_deployment(
            "node-a",
            identity("deployment-failed", 2, "model-failed", 'f'),
            assignment(),
            []() -> runtime::InferenceEngineParts {
                throw std::runtime_error("synthetic load failure");
            }
        );

    expect(!failed.ok, "throwing engine builder unexpectedly succeeded");
    expect(failed.error_code == "deployment_load_failed", "failed load used the wrong error");
    expect(manager.has_resident_deployment(), "failed load lost deployment identity");
    expect(
        manager.resident_deployment_state() == runtime::deployment::ResidentDeploymentState::Failed,
        "failed load did not enter failed state"
    );
    expect(!manager.has_active_deployment(), "failed load became active");

    const runtime::deployment::ActivateDeploymentResult activation =
        manager.activate_resident_deployment(identity("deployment-failed", 2, "model-failed", 'f'));
    expect(!activation.ok, "failed deployment activated");
    expect(activation.error_code == "deployment_failed", "failed activation used the wrong error");

    const runtime::deployment::UnloadDeploymentResult unloaded =
        manager.unload_resident_deployment(identity("deployment-failed", 2, "model-failed", 'f'));
    expect(unloaded.ok, "failed deployment could not be unloaded");
    expect(!manager.has_resident_deployment(), "failed deployment cleanup did not return idle");
}

} // namespace

int main() {
    test_prepare_activate_drain_handoff();
    test_failed_load_is_visible_and_recoverable_by_unload();

    std::cout << "runtime deployment lifecycle tests passed\n";
    return 0;
}
