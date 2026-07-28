#include "engine/generation_service.hpp"

#include "inference/stage.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime = jetsonfabric::runtime;

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

runtime::protocol::StageResponse response_for(
    const runtime::protocol::StageRequest& request
) {
    runtime::protocol::StageResponse response;
    response.session_id = request.session_id;
    response.request_id = request.request_id;
    response.model_id = request.model_id;
    response.deployment_id = request.deployment_id;
    response.deployment_epoch = request.deployment_epoch;
    response.model_sha256 = request.model_sha256;
    response.phase = request.phase;
    response.decode_step = request.decode_step;
    response.stage_index = request.stage_index;
    response.stage_count = request.stage_count;
    response.node_name = request.node_name;
    response.layer_start = request.layer_start;
    response.layer_end = request.layer_end;
    response.bytes_in = static_cast<std::int64_t>(request.payload.size());
    return response;
}

class RecordingLayerExecutor final : public runtime::pipeline_parallel::LayerExecutor {
public:
    runtime::inference::ExecutionResult execute(
        const runtime::inference::StageInput& input
    ) const override {
        ++execute_calls;
        last_request_id = input.request_id;

        runtime::inference::StageOutput output;
        output.payload.kind = runtime::inference::PayloadKind::Activation;
        output.payload.tensor.dtype = "f32";
        output.payload.tensor.shape = {1};
        output.payload.tensor.byte_order = "little";
        output.payload.tensor.layout = "row_major";
        output.payload.bytes = {0, 0, 0, 0};
        output.prompt_tokens =
            input.phase == runtime::inference::Phase::Prefill ? 4 : 0;
        return runtime::inference::ExecutionResult::success(std::move(output));
    }

    void close_session(const std::string& session_id) const override {
        ++close_calls;
        closed_session_id = session_id;
    }

    mutable int execute_calls = 0;
    mutable int close_calls = 0;
    mutable std::string last_request_id;
    mutable std::string closed_session_id;
};

class RecordingStageTransport final : public runtime::transport::StageTransport {
public:
    runtime::pipeline_parallel::StageRunResult invoke(
        const runtime::protocol::GenerationStage& stage,
        const runtime::protocol::StageRequest& request,
        runtime::pipeline_parallel::StageOperation operation
    ) const override {
        expect(stage.stage_index == 1, "transport received a local stage");

        runtime::pipeline_parallel::StageRunResult result;
        result.ok = true;
        result.status = "200 OK";
        result.response = response_for(request);
        if (operation == runtime::pipeline_parallel::StageOperation::CloseSession) {
            ++close_calls;
            result.response.payload_kind = "text";
            result.response.encoding = "utf-8";
            return result;
        }

        ++execute_calls;
        result.response.payload_kind = "sampled_token";
        result.response.encoding.clear();
        result.response.dtype = "u32";
        result.response.shape = {1};
        result.response.byte_order = "little";
        result.response.layout = "row_major";
        result.response.payload = {42, 0, 0, 0};
        result.response.bytes_out = 4;
        result.response.completion_tokens = 1;
        result.response.message = " answer";
        return result;
    }

    mutable int execute_calls = 0;
    mutable int close_calls = 0;
};

runtime::protocol::GenerationRequest generation_request() {
    return runtime::protocol::GenerationRequest{
        .request_id = "request-a",
        .session_id = "session-a",
        .model_id = "model-a",
        .prompt = "test prompt",
        .max_tokens = 1,
        .deployment = std::nullopt,
        .stages = {
            runtime::protocol::GenerationStage{
                .stage_index = 0,
                .stage_count = 2,
                .node_id = "node-a",
                .node_name = "dopey",
                .api_url = "http://dopey:52415",
                .layer_start = 0,
                .layer_end = 1,
            },
            runtime::protocol::GenerationStage{
                .stage_index = 1,
                .stage_count = 2,
                .node_id = "node-b",
                .node_name = "grumpy",
                .api_url = "http://grumpy:52415",
                .layer_start = 1,
                .layer_end = 2,
            },
        },
    };
}

void test_generation_dispatches_local_and_remote_stages() {
    auto executor = std::make_unique<RecordingLayerExecutor>();
    RecordingLayerExecutor* executor_observer = executor.get();
    runtime::deployment::ModelManager model_manager(
        "dopey",
        runtime::deployment::DeploymentIdentity{
            .deployment_id = "model-a",
            .epoch = 0,
            .model_id = "model-a",
            .model_sha256 = "",
        },
        runtime::pipeline_parallel::StageAssignment{
            .stage_index = 0,
            .stage_count = 2,
            .layer_start = 0,
            .layer_end = 1,
        },
        runtime::InferenceEngineParts{
            .layer_executor = std::move(executor),
            .model_residency = std::nullopt,
        }
    );
    RecordingStageTransport transport;
    runtime::GenerationService service(
        "dopey",
        runtime::ExecutionMode::PipelineParallel,
        model_manager,
        transport
    );

    std::vector<runtime::pipeline_parallel::GenerationToken> tokens;
    const runtime::pipeline_parallel::GenerationResult result = service.generate(
        generation_request(),
        [&tokens](const auto& token) {
            tokens.push_back(token);
            return true;
        }
    );

    expect(result.ok, "generation service failed");
    expect(result.sampled_tokens == std::vector<std::uint32_t>{42}, "sampled token changed");
    expect(result.prompt_tokens == 4, "prompt token accounting changed");
    expect(result.stage_calls == 2 && result.remote_stage_calls == 1, "stage routing changed");
    expect(tokens.size() == 1 && tokens.front().text == " answer", "token sink changed");
    expect(executor_observer->execute_calls == 1, "stage zero was not executed locally");
    expect(executor_observer->close_calls == 1, "local session was not closed");
    expect(transport.execute_calls == 1, "remote stage did not use the transport");
    expect(transport.close_calls == 1, "remote session was not closed");
}

void test_generation_rejects_wrong_stage_zero_node() {
    auto executor = std::make_unique<RecordingLayerExecutor>();
    runtime::deployment::ModelManager model_manager(
        "dopey",
        runtime::deployment::DeploymentIdentity{
            .deployment_id = "model-a",
            .epoch = 0,
            .model_id = "model-a",
            .model_sha256 = "",
        },
        runtime::pipeline_parallel::StageAssignment{
            .stage_index = 0,
            .stage_count = 2,
            .layer_start = 0,
            .layer_end = 1,
        },
        runtime::InferenceEngineParts{
            .layer_executor = std::move(executor),
            .model_residency = std::nullopt,
        }
    );
    RecordingStageTransport transport;
    runtime::GenerationService service(
        "dopey",
        runtime::ExecutionMode::PipelineParallel,
        model_manager,
        transport
    );
    auto request = generation_request();
    request.stages.front().node_name = "grumpy";

    const auto result = service.generate(request, [](const auto&) { return true; });

    expect(!result.ok, "generation accepted the wrong stage-zero node");
    expect(result.error_code == "invalid_pipeline_leader", "wrong validation error");
    expect(transport.execute_calls == 0, "invalid generation reached the transport");
}

} // namespace

int main() {
    try {
        test_generation_dispatches_local_and_remote_stages();
        test_generation_rejects_wrong_stage_zero_node();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "generation service tests passed\n";
    return 0;
}
