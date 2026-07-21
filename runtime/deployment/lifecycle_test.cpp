#include "deployment/model_manager.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

runtime::protocol::StageRequest request() {
    return runtime::protocol::StageRequest{
        .session_id = "session-lifecycle",
        .request_id = "request-lifecycle",
        .model_id = "model-a",
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

void test_idle_load_ready_activate_infer_unload_idle() {
    runtime::deployment::ModelManager manager;
    int build_calls = 0;
    int destruction_count = 0;
    RecordingExecutor* recording = nullptr;

    const runtime::deployment::LoadDeploymentResult loaded =
        manager.load_resident_deployment(
            "node-a",
            runtime::deployment::DeploymentIdentity{
                .deployment_id = "deployment-a",
                .model_id = "model-a",
            },
            assignment(),
            [&]() {
                ++build_calls;
                auto executor = std::make_unique<RecordingExecutor>(&destruction_count);
                recording = executor.get();
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
            runtime::deployment::DeploymentIdentity{
                .deployment_id = "deployment-b",
                .model_id = "model-b",
            },
            assignment(),
            [&]() {
                ++rejected_builder_calls;
                return runtime::InferenceEngineParts{};
            }
        );
    expect(!duplicate.ok, "runtime accepted a second resident deployment");
    expect(duplicate.error_code == "resident_deployment_exists", "duplicate load used the wrong error");
    expect(rejected_builder_calls == 0, "duplicate load built a second engine");

    const runtime::deployment::ActivateDeploymentResult stale =
        manager.activate_resident_deployment("deployment-b");
    expect(!stale.ok, "stale activation command succeeded");
    expect(stale.error_code == "deployment_mismatch", "stale activation used the wrong error");

    const runtime::deployment::ActivateDeploymentResult activated =
        manager.activate_resident_deployment("deployment-a");
    expect(activated.ok, "ready deployment failed to activate");
    expect(manager.has_active_deployment(), "activated deployment was not active");
    expect(
        manager.deployment_status().model_residency.has_value(),
        "activation lost model residency accounting"
    );

    const runtime::deployment::ActivateDeploymentResult repeated_activation =
        manager.activate_resident_deployment("deployment-a");
    expect(repeated_activation.ok, "activation retry was not idempotent");

    const runtime::pipeline_parallel::StageRunResult inference = manager.run_stage(request());
    expect(inference.ok, "active deployment failed inference");
    expect(recording != nullptr && recording->execute_calls == 1, "active executor was not called once");
    expect(recording->last_model_id == "model-a", "active executor received the wrong model");

    const runtime::deployment::UnloadDeploymentResult unloaded =
        manager.unload_resident_deployment("deployment-a");
    expect(unloaded.ok, "active deployment failed to unload");
    expect(destruction_count == 1, "unload did not release execution components exactly once");
    expect(!manager.has_resident_deployment(), "unload did not return the runtime to idle");
    expect(!manager.has_active_deployment(), "unload left an active deployment");
    expect(
        !manager.deployment_status().model_residency.has_value(),
        "unload retained model residency accounting"
    );
}

void test_failed_load_is_visible_and_recoverable_by_unload() {
    runtime::deployment::ModelManager manager;

    const runtime::deployment::LoadDeploymentResult failed =
        manager.load_resident_deployment(
            "node-a",
            runtime::deployment::DeploymentIdentity{
                .deployment_id = "deployment-failed",
                .model_id = "model-failed",
            },
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
        manager.activate_resident_deployment("deployment-failed");
    expect(!activation.ok, "failed deployment activated");
    expect(activation.error_code == "deployment_failed", "failed activation used the wrong error");

    const runtime::deployment::UnloadDeploymentResult unloaded =
        manager.unload_resident_deployment("deployment-failed");
    expect(unloaded.ok, "failed deployment could not be unloaded");
    expect(!manager.has_resident_deployment(), "failed deployment cleanup did not return idle");
}

} // namespace

int main() {
    test_idle_load_ready_activate_infer_unload_idle();
    test_failed_load_is_visible_and_recoverable_by_unload();

    std::cout << "runtime deployment lifecycle tests passed\n";
    return 0;
}
