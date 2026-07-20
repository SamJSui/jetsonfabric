#include "deployment/model_manager.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace runtime = jetsonfabric::runtime;

namespace {

using ResidentState = runtime::deployment::ResidentDeploymentState;
using OptionalResidentState = std::optional<ResidentState>;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void expect_transition(
    OptionalResidentState from,
    OptionalResidentState to,
    bool expected,
    const std::string& message
) {
    expect(
        runtime::deployment::is_valid_resident_deployment_transition(from, to) == expected,
        message
    );
}

class RecordingExecutor final : public runtime::pipeline_parallel::LayerExecutor {
public:
    runtime::inference::ExecutionResult execute(const runtime::inference::StageInput& input) const override {
        ++execute_calls;
        last_model_id = input.model_id;
        return runtime::inference::ExecutionResult::success(runtime::inference::StageOutput{
            .payload = runtime::inference::Payload{
                .kind = runtime::inference::PayloadKind::SampledToken,
                .encoding = "",
                .tensor = runtime::inference::TensorDescriptor{
                    .dtype = "u32",
                    .shape = {1},
                    .byte_order = "little",
                    .layout = "row_major",
                },
                .bytes = {42, 0, 0, 0},
            },
            .prompt_tokens = 2,
            .completion_tokens = 1,
        });
    }

    void close_session(const std::string& session_id) const override {
        ++close_calls;
        last_closed_session = session_id;
    }

    mutable int execute_calls = 0;
    mutable int close_calls = 0;
    mutable std::string last_model_id;
    mutable std::string last_closed_session;
};

runtime::pipeline_parallel::StageAssignment assignment() {
    return runtime::pipeline_parallel::StageAssignment{
        .stage_index = 0,
        .stage_count = 1,
        .layer_start = 0,
        .layer_end = 4,
    };
}

runtime::protocol::StageRequest valid_request() {
    return runtime::protocol::StageRequest{
        .session_id = "session-1",
        .request_id = "request-1",
        .model_id = "model-a",
        .phase = "prefill",
        .decode_step = 0,
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

void test_valid_resident_state_transitions() {
    expect_transition(std::nullopt, ResidentState::Loading, true, "idle must transition to loading");

    expect_transition(ResidentState::Loading, ResidentState::Ready, true, "loading must transition to ready");
    expect_transition(ResidentState::Loading, ResidentState::Failed, true, "loading must transition to failed");

    expect_transition(ResidentState::Ready, ResidentState::Active, true, "ready must transition to active");
    expect_transition(ResidentState::Ready, ResidentState::Unloading, true, "ready must transition to unloading");
    expect_transition(ResidentState::Ready, ResidentState::Failed, true, "ready must transition to failed");

    expect_transition(ResidentState::Active, ResidentState::Draining, true, "active must transition to draining");
    expect_transition(ResidentState::Active, ResidentState::Failed, true, "active must transition to failed");

    expect_transition(
        ResidentState::Draining,
        ResidentState::Unloading,
        true,
        "draining must transition to unloading"
    );
    expect_transition(ResidentState::Draining, ResidentState::Failed, true, "draining must transition to failed");

    expect_transition(ResidentState::Unloading, std::nullopt, true, "unloading must transition to idle");
    expect_transition(
        ResidentState::Unloading,
        ResidentState::Failed,
        true,
        "unloading must transition to failed"
    );

    expect_transition(ResidentState::Failed, ResidentState::Unloading, true, "failed must transition to unloading");
}

void test_invalid_resident_state_transitions() {
    expect_transition(std::nullopt, std::nullopt, false, "idle must not transition to idle");
    expect_transition(std::nullopt, ResidentState::Ready, false, "idle must not skip loading");
    expect_transition(ResidentState::Loading, ResidentState::Active, false, "loading must not skip ready");
    expect_transition(ResidentState::Ready, ResidentState::Draining, false, "ready must not enter draining");
    expect_transition(
        ResidentState::Active,
        ResidentState::Unloading,
        false,
        "active must drain before unloading"
    );
    expect_transition(ResidentState::Draining, ResidentState::Active, false, "draining must not reactivate");
    expect_transition(ResidentState::Ready, std::nullopt, false, "ready must unload before idle");
    expect_transition(ResidentState::Active, std::nullopt, false, "active must not transition directly to idle");
    expect_transition(ResidentState::Failed, ResidentState::Active, false, "failed must not reactivate");
    expect_transition(ResidentState::Failed, std::nullopt, false, "failed must unload before idle");

    constexpr std::array<ResidentState, 6> states{
        ResidentState::Loading,
        ResidentState::Ready,
        ResidentState::Active,
        ResidentState::Draining,
        ResidentState::Unloading,
        ResidentState::Failed,
    };
    for (const ResidentState state : states) {
        expect_transition(state, state, false, "resident state transitions must make progress");
    }
}

void test_idle_manager() {
    runtime::deployment::ModelManager manager;

    expect(!manager.has_resident_deployment(), "empty manager reported a resident deployment");
    expect(!manager.has_active_deployment(), "empty manager reported an active deployment");
    expect(
        manager.resident_deployment_identity() == nullptr,
        "empty manager exposed a resident deployment identity"
    );
    expect(
        !manager.resident_deployment_state().has_value(),
        "empty manager exposed a resident deployment state"
    );
    expect(manager.active_deployment_identity() == nullptr, "empty manager exposed an active identity");
    expect(manager.active_deployment_id().empty(), "empty manager reported an active deployment ID");
    expect(manager.active_model_id().empty(), "empty manager reported an active model identity");

    const runtime::pipeline_parallel::StageRunResult run_result = manager.run_stage(valid_request());
    expect(!run_result.ok, "empty manager accepted stage execution");
    expect(run_result.status == "503 Service Unavailable", "idle execution rejection used the wrong status");
    expect(run_result.error_code == "no_active_deployment", "idle execution rejection used the wrong error");

    const runtime::pipeline_parallel::StageRunResult close_result = manager.close_session(valid_request());
    expect(!close_result.ok, "empty manager accepted session close");
    expect(close_result.status == "503 Service Unavailable", "idle close rejection used the wrong status");
    expect(close_result.error_code == "no_active_deployment", "idle close rejection used the wrong error");
}

void test_loaded_manager() {
    auto executor = std::make_unique<RecordingExecutor>();
    RecordingExecutor* recording = executor.get();

    runtime::deployment::ModelManager manager(
        "node-a",
        runtime::deployment::DeploymentIdentity{
            .deployment_id = "deployment-a",
            .model_id = "model-a",
        },
        assignment(),
        runtime::InferenceEngineParts{.layer_executor = std::move(executor)}
    );

    expect(manager.has_resident_deployment(), "configured manager did not report a resident deployment");
    expect(manager.has_active_deployment(), "configured manager did not report an active deployment");

    const runtime::deployment::DeploymentIdentity* resident_identity =
        manager.resident_deployment_identity();
    expect(resident_identity != nullptr, "configured manager did not expose its resident identity");
    expect(resident_identity->deployment_id == "deployment-a", "resident deployment ID was not retained");
    expect(resident_identity->model_id == "model-a", "resident deployment model ID was not retained");
    expect(
        manager.resident_deployment_state() == ResidentState::Active,
        "configured deployment did not report the active resident state"
    );

    const runtime::deployment::DeploymentIdentity* active_identity =
        manager.active_deployment_identity();
    expect(active_identity != nullptr, "configured manager did not expose its active identity");
    expect(active_identity->deployment_id == "deployment-a", "active deployment ID was not retained");
    expect(active_identity->model_id == "model-a", "active deployment model ID was not retained");
    expect(manager.active_deployment_id() == "deployment-a", "active deployment ID query was incorrect");
    expect(manager.active_model_id() == "model-a", "active model identity was not retained");

    const runtime::pipeline_parallel::StageRunResult result = manager.run_stage(valid_request());
    expect(result.ok, "valid stage request did not reach the active deployment");
    expect(recording->execute_calls == 1, "active executor was not invoked exactly once");
    expect(recording->last_model_id == "model-a", "active executor received the wrong model identity");
    expect(result.response.payload_kind == "sampled_token", "active executor response was not returned");

    runtime::protocol::StageRequest wrong_model = valid_request();
    wrong_model.model_id = "model-b";
    const runtime::pipeline_parallel::StageRunResult rejected = manager.run_stage(wrong_model);
    expect(!rejected.ok, "request for an inactive model was accepted");
    expect(rejected.error_code == "invalid_stage_request", "inactive model rejection used the wrong error");
    expect(recording->execute_calls == 1, "rejected request reached the active executor");

    const runtime::pipeline_parallel::StageRunResult closed = manager.close_session(valid_request());
    expect(closed.ok, "session close did not reach the active deployment");
    expect(recording->close_calls == 1, "active executor did not receive session close");
    expect(recording->last_closed_session == "session-1", "wrong session was closed");
}

void test_invalid_identity_rejected() {
    const auto rejected = [](runtime::deployment::DeploymentIdentity identity) {
        try {
            runtime::deployment::ModelManager invalid(
                "node-a",
                std::move(identity),
                assignment(),
                runtime::InferenceEngineParts{
                    .layer_executor = std::make_unique<RecordingExecutor>(),
                }
            );
            (void) invalid;
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };

    expect(
        rejected(runtime::deployment::DeploymentIdentity{.deployment_id = "", .model_id = "model-a"}),
        "model manager accepted an empty deployment ID"
    );
    expect(
        rejected(runtime::deployment::DeploymentIdentity{.deployment_id = "deployment-a", .model_id = ""}),
        "model manager accepted an empty model ID"
    );
}

void test_missing_executor_rejected() {
    bool rejected_missing_executor = false;
    try {
        runtime::deployment::ModelManager invalid(
            "node-a",
            runtime::deployment::DeploymentIdentity{
                .deployment_id = "deployment-a",
                .model_id = "model-a",
            },
            assignment(),
            runtime::InferenceEngineParts{}
        );
        (void) invalid;
    } catch (const std::invalid_argument&) {
        rejected_missing_executor = true;
    }
    expect(rejected_missing_executor, "model manager accepted an empty engine deployment");
}

} // namespace

int main() {
    test_valid_resident_state_transitions();
    test_invalid_resident_state_transitions();
    test_idle_manager();
    test_loaded_manager();
    test_invalid_identity_rejected();
    test_missing_executor_rejected();

    std::cout << "runtime model manager tests passed\n";
    return 0;
}
