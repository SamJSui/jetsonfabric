#include "pipeline_parallel/generation_runner.hpp"

#include "protocol/stage.hpp"
#include "protocol/stage_control.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace runtime = jetsonfabric::runtime;

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

runtime::protocol::GenerationRequest generation_request(
    int max_tokens = 2,
    const std::string& suffix = "a"
) {
    return runtime::protocol::GenerationRequest{
        .request_id = "request-" + suffix,
        .session_id = "session-" + suffix,
        .model_id = "model-a",
        .prompt = "hello",
        .max_tokens = max_tokens,
        .deployment = std::nullopt,
        .stages = {
            runtime::protocol::GenerationStage{
                .stage_index = 0,
                .stage_count = 2,
                .node_id = "node-a",
                .node_name = "dopey",
                .api_url = "http://node-a:8080",
                .layer_start = 0,
                .layer_end = 4,
            },
            runtime::protocol::GenerationStage{
                .stage_index = 1,
                .stage_count = 2,
                .node_id = "node-b",
                .node_name = "sleepy",
                .api_url = "http://node-b:8080",
                .layer_start = 4,
                .layer_end = 8,
            },
        },
    };
}

runtime::protocol::StageResponse response_for(const runtime::protocol::StageRequest& request) {
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
    response.execution_us = 100;
    response.stage_total_us = 130;
    return response;
}

void set_u32_payload(runtime::protocol::StageResponse& response, std::uint32_t token) {
    response.payload_kind = "sampled_token";
    response.dtype = "u32";
    response.shape = {1};
    response.byte_order = "little";
    response.layout = "row_major";
    response.payload = {
        static_cast<std::uint8_t>(token & 0xffU),
        static_cast<std::uint8_t>((token >> 8U) & 0xffU),
        static_cast<std::uint8_t>((token >> 16U) & 0xffU),
        static_cast<std::uint8_t>((token >> 24U) & 0xffU),
    };
    response.bytes_out = 4;
}

struct InvocationHarness {
    int execute_calls = 0;
    int close_calls = 0;
    int fail_stage = -1;
    int stop_on_decode_step = -1;
    bool mismatch_cleanup_identity = false;
    std::vector<std::string> execute_request_ids;
    std::vector<std::string> token_texts;

    runtime::pipeline_parallel::StageRunResult invoke(
        const runtime::protocol::GenerationStage& stage,
        const runtime::protocol::StageRequest& request,
        runtime::pipeline_parallel::StageOperation operation
    ) {
        runtime::pipeline_parallel::StageRunResult result;
        if (operation == runtime::pipeline_parallel::StageOperation::CloseSession) {
            ++close_calls;
            result.ok = true;
            result.status = "200 OK";
            result.response = response_for(request);
            result.response.payload_kind = "text";
            result.response.encoding = "utf-8";
            if (mismatch_cleanup_identity && close_calls == 1) {
                result.response.deployment_epoch += 1;
            }
            return result;
        }
        ++execute_calls;
        execute_request_ids.push_back(request.request_id);
        if (request.payload_kind != "text" && !request.encoding.empty()) {
            result.status = "400 Bad Request";
            result.error_code = "invalid_tensor_encoding";
            result.error_message = "tensor payload declared a text encoding";
            return result;
        }
        if (stage.stage_index == fail_stage) {
            result.status = "502 Bad Gateway";
            result.error_code = "injected_stage_failure";
            result.error_message = "injected failure";
            return result;
        }
        result.ok = true;
        result.status = "200 OK";
        result.response = response_for(request);
        if (stage.stage_index != 0) result.remote_call_us = 200;
        if (stage.stage_index == 0) {
            result.response.payload_kind = "activation";
            result.response.encoding.clear();
            result.response.dtype = "f32";
            result.response.shape = {1};
            result.response.byte_order = "little";
            result.response.layout = "row_major";
            result.response.payload = {1, 2, 3, 4};
            result.response.bytes_out = 4;
            if (request.phase == "prefill") result.response.prompt_tokens = 5;
        } else {
            set_u32_payload(result.response, static_cast<std::uint32_t>(41 + request.decode_step));
            if (request.decode_step == stop_on_decode_step) {
                result.response.completion_tokens = 0;
            } else {
                result.response.message = token_texts.empty()
                    ? (request.decode_step == 0 ? "hello" : " world")
                    : token_texts.at(static_cast<std::size_t>(request.decode_step));
                result.response.completion_tokens = 1;
            }
        }
        return result;
    }
};

struct ConcurrentInvocationHarness {
    runtime::pipeline_parallel::StageRunResult invoke(
        const runtime::protocol::GenerationStage& stage,
        const runtime::protocol::StageRequest& request,
        runtime::pipeline_parallel::StageOperation operation
    ) {
        runtime::pipeline_parallel::StageRunResult result;
        result.ok = true;
        result.status = "200 OK";
        result.response = response_for(request);
        if (operation == runtime::pipeline_parallel::StageOperation::CloseSession) {
            result.response.payload_kind = "text";
            result.response.encoding = "utf-8";
            return result;
        }

        const std::size_t stage_index = static_cast<std::size_t>(stage.stage_index);
        std::lock_guard stage_lock(stage_mutex.at(stage_index));
        active.at(stage_index).store(true);
        if (active.at(1U - stage_index).load()) overlap_observed.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        active.at(stage_index).store(false);

        if (stage.stage_index == 0) {
            result.response.payload_kind = "activation";
            result.response.dtype = "f32";
            result.response.shape = {1};
            result.response.byte_order = "little";
            result.response.layout = "row_major";
            result.response.payload = {1, 2, 3, 4};
            if (request.phase == "prefill") result.response.prompt_tokens = 1;
        } else {
            set_u32_payload(result.response, static_cast<std::uint32_t>(41 + request.decode_step));
            result.response.message = "x";
            result.response.completion_tokens = 1;
        }
        return result;
    }

    std::array<std::mutex, 2> stage_mutex;
    std::array<std::atomic_bool, 2> active{false, false};
    std::atomic_bool overlap_observed{false};
};

void test_generation_owns_both_loops_and_cleanup() {
    InvocationHarness harness;
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& request, auto operation) {
        return harness.invoke(stage, request, operation);
    });
    std::vector<runtime::pipeline_parallel::GenerationToken> emitted;
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        generation_request(),
        [&emitted](const auto& token) {
            emitted.push_back(token);
            return true;
        }
    );
    expect(result.ok, "generation runner failed");
    expect(result.finish_reason == "length", "generation runner used the wrong finish reason");
    expect(result.sampled_tokens == std::vector<std::uint32_t>({41, 42}), "sampled tokens changed");
    expect(result.prompt_tokens == 5 && result.completion_tokens == 2, "usage accounting changed");
    expect(result.stage_calls == 4 && result.remote_stage_calls == 2, "stage call accounting changed");
    expect(result.stage_timings.size() == 4, "stage timings were not aggregated by phase and stage");
    expect(
        result.stage_timings[0].phase == "prefill" &&
            result.stage_timings[0].stage_index == 0 &&
            result.stage_timings[0].execution_us == 100 &&
            result.stage_timings[0].remote_call_us == 0,
        "local prefill timing changed"
    );
    expect(
        result.stage_timings[1].phase == "prefill" &&
            result.stage_timings[1].stage_index == 1 &&
            result.stage_timings[1].remote_call_us == 200 &&
            result.stage_timings[1].remote_overhead_us == 70,
        "remote prefill timing changed"
    );
    expect(harness.execute_calls == 4 && harness.close_calls == 2, "runner did not execute and close every stage");
    expect(
        harness.execute_request_ids == std::vector<std::string>({
            "request-a-prefill-0-stage-0",
            "request-a-prefill-0-stage-1",
            "request-a-decode-1-stage-0",
            "request-a-decode-1-stage-1",
        }),
        "stage request IDs were not independently derived from the pass request"
    );
    expect(emitted.size() == 2 && emitted[0].text == "hello" && emitted[1].text == " world", "token stream changed");
}

void test_independent_generations_overlap_pipeline_stages() {
    ConcurrentInvocationHarness harness;
    std::promise<void> start_signal;
    const std::shared_future<void> start = start_signal.get_future().share();
    auto run = [&harness, start](const std::string& suffix) {
        return std::async(std::launch::async, [&harness, start, suffix]() {
            start.wait();
            runtime::pipeline_parallel::GenerationRunner runner([
                &harness
            ](const auto& stage, const auto& request, auto operation) {
                return harness.invoke(stage, request, operation);
            });
            return runner.run(generation_request(3, suffix), [](const auto&) {
                return true;
            });
        });
    };

    std::future<runtime::pipeline_parallel::GenerationResult> first = run("a");
    std::future<runtime::pipeline_parallel::GenerationResult> second = run("b");
    start_signal.set_value();
    const runtime::pipeline_parallel::GenerationResult first_result = first.get();
    const runtime::pipeline_parallel::GenerationResult second_result = second.get();

    expect(first_result.ok && second_result.ok, "concurrent generation failed");
    expect(
        harness.overlap_observed.load(),
        "independent generations did not overlap work on separate pipeline stages"
    );
}

void test_sink_cancellation_closes_every_stage() {
    InvocationHarness harness;
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& request, auto operation) {
        return harness.invoke(stage, request, operation);
    });
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        generation_request(4),
        [](const auto&) { return false; }
    );
    expect(!result.ok && result.error_code == "generation_canceled", "sink cancellation was not reported");
    expect(harness.execute_calls == 2 && harness.close_calls == 2, "cancellation did not close every stage");
}

void test_generation_coalesces_split_utf8_token_bytes() {
    InvocationHarness harness;
    harness.token_texts = {
        std::string("prefix ") + static_cast<char>(0xe4),
        std::string(1, static_cast<char>(0xbd)),
        std::string(1, static_cast<char>(0xa0)),
    };
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& request, auto operation) {
        return harness.invoke(stage, request, operation);
    });
    std::vector<runtime::pipeline_parallel::GenerationToken> emitted;
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        generation_request(3),
        [&emitted](const auto& token) {
            emitted.push_back(token);
            return true;
        }
    );

    expect(result.ok, "generation rejected a split UTF-8 token");
    expect(emitted.size() == 3, "split UTF-8 changed token event accounting");
    expect(emitted[0].text == "prefix ", "complete UTF-8 prefix was not emitted");
    expect(emitted[1].text.empty(), "incomplete UTF-8 continuation was emitted");
    expect(emitted[2].text == "\xe4\xbd\xa0", "split UTF-8 bytes were not coalesced exactly");
}

void test_natural_stop_excludes_eos_and_accounts_for_its_pass() {
    InvocationHarness harness;
    harness.stop_on_decode_step = 1;
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& request, auto operation) {
        return harness.invoke(stage, request, operation);
    });
    std::vector<runtime::pipeline_parallel::GenerationToken> emitted;
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        generation_request(4),
        [&emitted](const auto& token) {
            emitted.push_back(token);
            return true;
        }
    );
    expect(result.ok && result.finish_reason == "stop", "natural stop was not reported");
    expect(result.sampled_tokens == std::vector<std::uint32_t>({41}), "EOS leaked into sampled tokens");
    expect(result.completion_tokens == 1, "natural stop completion count changed");
    expect(result.stage_calls == 4 && result.remote_stage_calls == 2, "EOS pass was not accounted for");
    expect(emitted.size() == 1 && emitted[0].token == 41, "EOS leaked into the token stream");
    expect(harness.close_calls == 2, "natural stop did not close every stage");
}

void test_stage_failure_closes_every_stage() {
    InvocationHarness harness;
    harness.fail_stage = 1;
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& request, auto operation) {
        return harness.invoke(stage, request, operation);
    });
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        generation_request(),
        [](const auto&) { return true; }
    );
    expect(!result.ok && result.error_code == "injected_stage_failure", "stage failure was not preserved");
    expect(harness.close_calls == 2, "stage failure did not close every stage");
}

void test_generation_propagates_managed_deployment_identity() {
    InvocationHarness harness;
    runtime::protocol::GenerationRequest request = generation_request(1);
    request.deployment = runtime::deployment::DeploymentIdentity{
        .deployment_id = "deployment-a",
        .epoch = 7,
        .model_id = "model-a",
        .model_sha256 = std::string(64, 'a'),
    };
    int identity_calls = 0;
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness,
        &identity_calls
    ](const auto& stage, const auto& stage_request, auto operation) {
        expect(stage_request.deployment_id == "deployment-a", "stage request omitted deployment ID");
        expect(stage_request.deployment_epoch == 7, "stage request omitted deployment epoch");
        expect(stage_request.model_sha256 == std::string(64, 'a'), "stage request omitted model hash");
        ++identity_calls;
        return harness.invoke(stage, stage_request, operation);
    });
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        request,
        [](const auto&) { return true; }
    );
    expect(result.ok, "managed generation failed");
    expect(identity_calls == 4, "managed identity did not reach execute and cleanup calls");
}

void test_cleanup_rejects_mismatched_success_identity() {
    InvocationHarness harness;
    harness.mismatch_cleanup_identity = true;
    runtime::protocol::GenerationRequest request = generation_request(1);
    request.deployment = runtime::deployment::DeploymentIdentity{
        .deployment_id = "deployment-a",
        .epoch = 7,
        .model_id = "model-a",
        .model_sha256 = std::string(64, 'a'),
    };
    runtime::pipeline_parallel::GenerationRunner runner([
        &harness
    ](const auto& stage, const auto& stage_request, auto operation) {
        return harness.invoke(stage, stage_request, operation);
    });
    const runtime::pipeline_parallel::GenerationResult result = runner.run(
        request,
        [](const auto&) { return true; }
    );
    expect(!result.ok, "mismatched cleanup response was accepted");
    expect(result.error_code == "generation_cleanup_failed", "cleanup identity failure used the wrong error code");
    expect(harness.close_calls == 2, "cleanup identity failure did not attempt every stage");
}

void test_generation_protocol_and_stagewire_round_trip() {
    const std::string body = R"({
        "request_id":"request-a",
        "session_id":"session-a",
        "model_id":"model-a",
        "prompt":"hello",
        "max_tokens":2,
        "deployment":{"deployment_id":"deployment-a","epoch":7,"model_id":"model-a","model_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        "stages":[
          {"stage_index":0,"stage_count":1,"node_id":"node-a","node_name":"dopey","api_url":"http://node-a:8080","layer_start":0,"layer_end":8}
        ]
    })";
    const runtime::protocol::GenerationRequest decoded = runtime::protocol::decode_generation_request(body);
    expect(decoded.stages.size() == 1 && decoded.max_tokens == 2, "generation request did not decode");
    expect(
        decoded.deployment.has_value() && decoded.deployment->deployment_id == "deployment-a" &&
            decoded.deployment->epoch == 7 && decoded.deployment->model_sha256 == std::string(64, 'a'),
        "generation deployment identity did not decode"
    );

    runtime::protocol::StageRequest request;
    request.session_id = "session-a";
    request.request_id = "request-a";
    request.model_id = "model-a";
    request.deployment_id = "deployment-a";
    request.deployment_epoch = 7;
    request.model_sha256 = std::string(64, 'a');
    request.stage_index = 0;
    request.stage_count = 1;
    request.node_name = "dopey";
    request.layer_end = 8;
    request.payload_kind = "text";
    request.encoding = "utf-8";
    request.payload = {'h', 'i'};
    const std::string frame = runtime::protocol::encode_stage_request(
        request,
        runtime::protocol::kStageOperationExecute
    );
    const runtime::protocol::StageRequest request_round_trip = runtime::protocol::decode_stage_request(frame);
    expect(request_round_trip.payload == request.payload, "stage request payload changed during round trip");
    expect(
        request_round_trip.deployment_id == request.deployment_id &&
            request_round_trip.deployment_epoch == request.deployment_epoch &&
            request_round_trip.model_sha256 == request.model_sha256,
        "stage request deployment identity changed during round trip"
    );
    expect(
        runtime::protocol::decode_stage_operation(frame) == runtime::protocol::kStageOperationExecute,
        "stage request operation changed during round trip"
    );

    runtime::protocol::StageResponse response = response_for(request_round_trip);
    set_u32_payload(response, 73);
    response.activation_decode_us = 11;
    response.activation_encode_us = 12;
    response.message = std::string(1, static_cast<char>(0x9e));
    const runtime::protocol::StageResponse response_round_trip = runtime::protocol::decode_stage_response(
        runtime::protocol::encode_stage_response(response)
    );
    expect(response_round_trip.payload == response.payload, "stage response payload changed during round trip");
    expect(response_round_trip.message == response.message, "stage response token bytes changed during round trip");
    expect(
        response_round_trip.execution_us == response.execution_us &&
            response_round_trip.activation_decode_us == response.activation_decode_us &&
            response_round_trip.activation_encode_us == response.activation_encode_us &&
            response_round_trip.stage_total_us == response.stage_total_us,
        "stage response timings changed during round trip"
    );
    expect(
        response_round_trip.deployment_id == request.deployment_id &&
            response_round_trip.deployment_epoch == request.deployment_epoch &&
            response_round_trip.model_sha256 == request.model_sha256,
        "stage response deployment identity changed during round trip"
    );
}

void test_generation_protocol_rejects_inconsistent_plan() {
    const std::string body = R"({
        "request_id":"request-a",
        "session_id":"session-a",
        "model_id":"model-a",
        "prompt":"hello",
        "max_tokens":2,
        "stages":[
          {"stage_index":1,"stage_count":1,"node_id":"node-a","node_name":"dopey","api_url":"http://node-a:8080","layer_start":0,"layer_end":8}
        ]
    })";
    bool rejected = false;
    try {
        (void) runtime::protocol::decode_generation_request(body);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "inconsistent generation plan was accepted");
}

void test_generation_protocol_rejects_empty_prompt() {
    const std::string body = R"({
        "request_id":"request-a",
        "session_id":"session-a",
        "model_id":"model-a",
        "prompt":"",
        "max_tokens":2,
        "stages":[
          {"stage_index":0,"stage_count":1,"node_id":"node-a","node_name":"dopey","api_url":"http://node-a:8080","layer_start":0,"layer_end":8}
        ]
    })";
    bool rejected = false;
    try {
        (void) runtime::protocol::decode_generation_request(body);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "empty generation prompt was accepted");
}

} // namespace

int main() {
    try {
        test_generation_owns_both_loops_and_cleanup();
        test_independent_generations_overlap_pipeline_stages();
        test_sink_cancellation_closes_every_stage();
        test_generation_coalesces_split_utf8_token_bytes();
        test_natural_stop_excludes_eos_and_accounts_for_its_pass();
        test_stage_failure_closes_every_stage();
        test_generation_propagates_managed_deployment_identity();
        test_cleanup_rejects_mismatched_success_identity();
        test_generation_protocol_and_stagewire_round_trip();
        test_generation_protocol_rejects_inconsistent_plan();
        test_generation_protocol_rejects_empty_prompt();
        std::cout << "generation runner tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
