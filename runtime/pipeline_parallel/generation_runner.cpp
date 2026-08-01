#include "pipeline_parallel/generation_runner.hpp"

#include "inference/stage.hpp"
#include "protocol/utf8.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::pipeline_parallel {
namespace {

GenerationResult generation_error(std::string status, std::string code, std::string message) {
    GenerationResult result;
    result.status = std::move(status);
    result.error_code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

void write_u32_le(std::vector<std::uint8_t>& payload, std::uint32_t value) {
    payload.push_back(static_cast<std::uint8_t>(value & 0xffU));
    payload.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    payload.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xffU));
    payload.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xffU));
}

struct SampledBatch {
    std::vector<std::uint32_t> tokens;
    std::vector<std::string> text;
    std::vector<bool> end_of_generation;
};

std::uint32_t read_u32_le(const std::uint8_t* payload) {
    return static_cast<std::uint32_t>(payload[0]) |
           (static_cast<std::uint32_t>(payload[1]) << 8U) |
           (static_cast<std::uint32_t>(payload[2]) << 16U) |
           (static_cast<std::uint32_t>(payload[3]) << 24U);
}

SampledBatch read_sampled_batch(const protocol::StageResponse& response) {
    if (response.payload_kind == "sampled_token") {
        if (response.payload.size() != 4U) {
            throw std::invalid_argument("final stage returned an invalid sampled_token");
        }
        return SampledBatch{
            .tokens = {read_u32_le(response.payload.data())},
            .text = {response.message},
            .end_of_generation = {response.completion_tokens == 0},
        };
    }
    if (response.payload_kind != "sampled_tokens" ||
        response.payload.size() < 8U || response.payload.size() % 4U != 0U) {
        throw std::invalid_argument("final stage did not return sampled token output");
    }

    const std::size_t count = response.payload.size() / 4U;
    if (response.token_text_offsets.size() != count || response.token_eog.size() != count) {
        throw std::invalid_argument("sampled_tokens metadata does not match its payload");
    }
    SampledBatch batch;
    batch.tokens.reserve(count);
    batch.text.reserve(count);
    batch.end_of_generation.reserve(count);
    std::size_t text_start = 0;
    for (std::size_t index = 0; index < count; ++index) {
        batch.tokens.push_back(read_u32_le(response.payload.data() + index * 4U));
        const std::size_t text_end = response.token_text_offsets[index];
        if (text_end < text_start || text_end > response.message.size()) {
            throw std::invalid_argument("sampled_tokens text offsets are invalid");
        }
        batch.text.push_back(response.message.substr(text_start, text_end - text_start));
        batch.end_of_generation.push_back(response.token_eog[index] != 0U);
        text_start = text_end;
    }
    if (text_start != response.message.size()) {
        throw std::invalid_argument("sampled_tokens text offsets do not consume the message");
    }
    return batch;
}

protocol::StageRequest initial_request(
    const protocol::GenerationRequest& request,
    const std::string& phase,
    int decode_step,
    const std::vector<std::uint32_t>& decode_tokens
) {
    protocol::StageRequest stage_request;
    stage_request.session_id = request.session_id;
    stage_request.request_id = request.request_id + "-" + phase + "-" + std::to_string(decode_step);
    stage_request.model_id = request.model_id;
    if (request.deployment.has_value()) {
        stage_request.deployment_id = request.deployment->deployment_id;
        stage_request.deployment_epoch = request.deployment->epoch;
        stage_request.model_sha256 = request.deployment->model_sha256;
    }
    stage_request.phase = phase;
    stage_request.decode_step = decode_step;
    stage_request.max_tokens = request.max_tokens;
    if (phase == "prefill") {
        stage_request.payload_kind = "text";
        stage_request.encoding = "utf-8";
        stage_request.payload.assign(request.prompt.begin(), request.prompt.end());
    } else {
        stage_request.payload_kind = decode_tokens.size() == 1U ? "sampled_token" : "tokens";
        stage_request.encoding.clear();
        stage_request.dtype = "u32";
        stage_request.shape = {static_cast<std::int64_t>(decode_tokens.size())};
        stage_request.byte_order = "little";
        stage_request.layout = "row_major";
        for (const std::uint32_t token : decode_tokens) {
            write_u32_le(stage_request.payload, token);
        }
    }
    return stage_request;
}

void apply_stage(const protocol::GenerationStage& stage, protocol::StageRequest& request) {
    request.request_id += "-stage-" + std::to_string(stage.stage_index);
    request.stage_index = stage.stage_index;
    request.stage_count = stage.stage_count;
    request.node_name = stage.node_name;
    request.layer_start = stage.layer_start;
    request.layer_end = stage.layer_end;
}

void apply_response(protocol::StageRequest& request, const protocol::StageResponse& response) {
    request.payload_kind = response.payload_kind;
    request.encoding = response.encoding;
    request.dtype = response.dtype;
    request.shape = response.shape;
    request.byte_order = response.byte_order;
    request.layout = response.layout;
    request.payload = response.payload;
}

std::string validate_response_identity(
    const protocol::StageRequest& request,
    const protocol::StageResponse& response
) {
    if (response.session_id != request.session_id ||
        response.request_id != request.request_id ||
        response.model_id != request.model_id ||
        response.deployment_id != request.deployment_id ||
        response.deployment_epoch != request.deployment_epoch ||
        response.model_sha256 != request.model_sha256 ||
        response.phase != request.phase ||
        response.decode_step != request.decode_step ||
        response.stage_index != request.stage_index ||
        response.stage_count != request.stage_count ||
        response.node_name != request.node_name ||
        response.layer_start != request.layer_start ||
        response.layer_end != request.layer_end) {
        return "stage response identity does not match its request";
    }
    return "";
}

std::string validate_response(
    const protocol::StageRequest& request,
    const protocol::StageResponse& response
) {
    const std::string identity_error = validate_response_identity(request, response);
    if (!identity_error.empty()) return identity_error;
    try {
        const inference::PayloadKind actual = inference::parse_payload_kind(response.payload_kind);
        const inference::PayloadKind expected = inference::expected_output(
            inference::parse_phase(request.phase),
            inference::StagePosition{.index = request.stage_index, .count = request.stage_count}
        );
        const bool speculative_final_output =
            request.phase == "decode" &&
            request.stage_count > 0 && request.stage_index == request.stage_count - 1 &&
            !request.shape.empty() && request.shape[0] > 1 &&
            actual == inference::PayloadKind::SampledTokens;
        if (actual != expected && !speculative_final_output) {
            return "stage response violates the pipeline payload transition";
        }
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

void record_stage_timing(
    GenerationResult& result,
    const protocol::GenerationStage& stage,
    const StageRunResult& run
) {
    protocol::GenerationStageTiming* aggregate = nullptr;
    for (protocol::GenerationStageTiming& timing : result.stage_timings) {
        if (timing.phase == run.response.phase && timing.stage_index == stage.stage_index) {
            aggregate = &timing;
            break;
        }
    }
    if (aggregate == nullptr) {
        result.stage_timings.push_back(protocol::GenerationStageTiming{
            .phase = run.response.phase,
            .stage_index = stage.stage_index,
            .node_name = stage.node_name,
            .remote = stage.stage_index != 0,
        });
        aggregate = &result.stage_timings.back();
    }

    const std::int64_t overhead = run.remote_call_us > run.response.stage_total_us
        ? run.remote_call_us - run.response.stage_total_us
        : 0;
    ++aggregate->calls;
    aggregate->batch_items += run.response.execution_batch_size;
    aggregate->max_execution_batch_size = std::max(
        aggregate->max_execution_batch_size,
        run.response.execution_batch_size
    );
    aggregate->verification_items += run.response.verification_width;
    aggregate->max_verification_width = std::max(
        aggregate->max_verification_width,
        run.response.verification_width
    );
    aggregate->execution_us += run.response.execution_us;
    aggregate->activation_decode_us += run.response.activation_decode_us;
    aggregate->activation_encode_us += run.response.activation_encode_us;
    aggregate->stage_total_us += run.response.stage_total_us;
    aggregate->remote_call_us += run.remote_call_us;
    aggregate->remote_overhead_us += overhead;
    aggregate->bytes_in += run.response.bytes_in;
    aggregate->bytes_out += run.response.bytes_out;
}

struct PassResult {
    bool ok = false;
    GenerationResult error;
    protocol::StageResponse final_response;
    std::vector<std::uint32_t> prompt_token_ids;
};

PassResult failed_pass(GenerationResult error) {
    PassResult result;
    result.error = std::move(error);
    return result;
}

PassResult run_pass(
    const protocol::GenerationRequest& generation,
    const StageInvoker& invoke_stage,
    protocol::StageRequest request,
    GenerationResult& result
) {
    protocol::StageResponse final_response;
    std::vector<std::uint32_t> prompt_token_ids;
    const std::string pass_request_id = request.request_id;
    for (const protocol::GenerationStage& stage : generation.stages) {
        request.request_id = pass_request_id;
        apply_stage(stage, request);
        StageRunResult stage_result = invoke_stage(stage, request, StageOperation::Execute);
        ++result.stage_calls;
        if (stage.stage_index != 0) ++result.remote_stage_calls;
        if (!stage_result.ok) {
            std::ostringstream message;
            message << "stage " << stage.stage_index << ": " << stage_result.error_message;
            return failed_pass(generation_error(
                stage_result.status,
                stage_result.error_code.empty() ? "generation_stage_failed" : stage_result.error_code,
                message.str()
            ));
        }
        const std::string response_error = validate_response(request, stage_result.response);
        if (!response_error.empty()) {
            return failed_pass(generation_error(
                "502 Bad Gateway", "invalid_stage_response", response_error
            ));
        }
        record_stage_timing(result, stage, stage_result);
        result.prompt_tokens += stage_result.response.prompt_tokens;
        if (!stage_result.response.prompt_token_ids.empty()) {
            prompt_token_ids = stage_result.response.prompt_token_ids;
        }
        result.bytes_in += stage_result.response.bytes_in;
        result.bytes_out += stage_result.response.bytes_out;
        final_response = stage_result.response;
        apply_response(request, stage_result.response);
    }
    return PassResult{true, {}, std::move(final_response), std::move(prompt_token_ids)};
}

std::string close_sessions(const protocol::GenerationRequest& generation, const StageInvoker& invoke_stage) {
    std::string first_error;
    for (const protocol::GenerationStage& stage : generation.stages) {
        protocol::StageRequest request;
        request.session_id = generation.session_id;
        request.request_id = generation.request_id + "-close-stage-" + std::to_string(stage.stage_index);
        request.model_id = generation.model_id;
        if (generation.deployment.has_value()) {
            request.deployment_id = generation.deployment->deployment_id;
            request.deployment_epoch = generation.deployment->epoch;
            request.model_sha256 = generation.deployment->model_sha256;
        }
        request.phase = "prefill";
        request.stage_index = stage.stage_index;
        request.stage_count = stage.stage_count;
        request.node_name = stage.node_name;
        request.layer_start = stage.layer_start;
        request.layer_end = stage.layer_end;
        request.payload_kind = "text";
        request.encoding = "utf-8";
        request.max_tokens = 1;
        StageRunResult close = invoke_stage(stage, request, StageOperation::CloseSession);
        if (!close.ok && first_error.empty()) {
            first_error = "stage " + std::to_string(stage.stage_index) + ": " + close.error_message;
        } else if (close.ok) {
            const std::string response_error = validate_response_identity(request, close.response);
            if (!response_error.empty() && first_error.empty()) {
                first_error = "stage " + std::to_string(stage.stage_index) + ": " + response_error;
            }
        }
    }
    return first_error;
}

std::string rollback_sessions(
    const protocol::GenerationRequest& generation,
    const StageInvoker& invoke_stage,
    int token_count,
    GenerationResult& result
) {
    std::string first_error;
    for (const protocol::GenerationStage& stage : generation.stages) {
        protocol::StageRequest request;
        request.session_id = generation.session_id;
        request.request_id = generation.request_id + "-rollback-stage-" +
            std::to_string(stage.stage_index);
        request.model_id = generation.model_id;
        if (generation.deployment.has_value()) {
            request.deployment_id = generation.deployment->deployment_id;
            request.deployment_epoch = generation.deployment->epoch;
            request.model_sha256 = generation.deployment->model_sha256;
        }
        request.phase = "decode";
        request.decode_step = 1;
        request.stage_index = stage.stage_index;
        request.stage_count = stage.stage_count;
        request.node_name = stage.node_name;
        request.layer_start = stage.layer_start;
        request.layer_end = stage.layer_end;
        request.payload_kind = "sampled_token";
        request.dtype = "u32";
        request.shape = {1};
        request.byte_order = "little";
        request.layout = "row_major";
        request.max_tokens = 1;
        request.rollback_tokens = token_count;
        write_u32_le(request.payload, 0);

        StageRunResult rollback = invoke_stage(stage, request, StageOperation::RollbackSession);
        ++result.rollback_stage_calls;
        result.rollback_stage_us += rollback.response.stage_total_us;
        result.rollback_remote_call_us += rollback.remote_call_us;
        if (stage.stage_index != 0) ++result.remote_rollback_stage_calls;
        if (!rollback.ok && first_error.empty()) {
            first_error = "stage " + std::to_string(stage.stage_index) + ": " +
                rollback.error_message;
        } else if (rollback.ok) {
            const std::string response_error = validate_response_identity(request, rollback.response);
            if (!response_error.empty() && first_error.empty()) {
                first_error = "stage " + std::to_string(stage.stage_index) + ": " + response_error;
            }
        }
    }
    return first_error;
}

} // namespace

GenerationRunner::GenerationRunner(
    StageInvoker invoke_stage,
    std::shared_ptr<const speculative::DraftStrategy> draft_strategy,
    int speculative_max_tokens
)
    : invoke_stage_(std::move(invoke_stage)),
      draft_strategy_(std::move(draft_strategy)),
      speculative_max_tokens_(speculative_max_tokens) {
    if (!invoke_stage_) {
        throw std::invalid_argument("generation runner requires a stage invoker");
    }
    if (speculative_max_tokens_ <= 0) {
        throw std::invalid_argument("speculative max tokens must be greater than zero");
    }
}

GenerationResult GenerationRunner::run(
    const protocol::GenerationRequest& request,
    const TokenSink& sink
) const {
    if (!sink) {
        return generation_error("500 Internal Server Error", "generation_sink_missing", "token sink is required");
    }

    GenerationResult result;
    bool end_of_generation = false;
    std::uint32_t previous_token = 0;
    int decode_step = 1;
    bool recover_after_rejection = false;
    std::string pending_text;
    std::vector<std::uint32_t> history;

    const auto fail = [&](std::string status, std::string code, std::string message) {
        const std::string cleanup_error = close_sessions(request, invoke_stage_);
        if (!cleanup_error.empty()) message += "; cleanup: " + cleanup_error;
        return generation_error(std::move(status), std::move(code), std::move(message));
    };
    const auto emit = [&](std::uint32_t token, const std::string& text) -> std::optional<GenerationResult> {
        pending_text += text;
        std::string emitted_text;
        try {
            const std::size_t complete_bytes = protocol::complete_utf8_prefix(pending_text);
            emitted_text = pending_text.substr(0, complete_bytes);
            pending_text.erase(0, complete_bytes);
        } catch (const std::invalid_argument& error) {
            return fail("502 Bad Gateway", "invalid_token_text", error.what());
        }
        const int token_index = static_cast<int>(result.sampled_tokens.size());
        result.sampled_tokens.push_back(token);
        history.push_back(token);
        ++result.completion_tokens;
        previous_token = token;
        if (!sink(GenerationToken{token, std::move(emitted_text), token_index})) {
            return fail(
                "499 Client Closed Request",
                "generation_canceled",
                "generation token sink canceled the request"
            );
        }
        return std::nullopt;
    };

    protocol::StageRequest prefill_request = initial_request(request, "prefill", 0, {});
    PassResult prefill = run_pass(request, invoke_stage_, std::move(prefill_request), result);
    if (!prefill.ok) {
        const std::string cleanup_error = close_sessions(request, invoke_stage_);
        if (!cleanup_error.empty()) prefill.error.error_message += "; cleanup: " + cleanup_error;
        return prefill.error;
    }
    SampledBatch prefill_output;
    try {
        prefill_output = read_sampled_batch(prefill.final_response);
    } catch (const std::exception& error) {
        return fail("502 Bad Gateway", "invalid_sampled_token", error.what());
    }
    if (prefill_output.tokens.size() != 1U) {
        return fail("502 Bad Gateway", "invalid_sampled_token", "prefill returned multiple tokens");
    }
    history = std::move(prefill.prompt_token_ids);
    end_of_generation = prefill_output.end_of_generation.front();
    if (!end_of_generation) {
        if (auto error = emit(prefill_output.tokens.front(), prefill_output.text.front());
            error.has_value()) {
            return *error;
        }
    }

    while (result.completion_tokens < request.max_tokens && !end_of_generation) {
        const int remaining = request.max_tokens - result.completion_tokens;
        std::vector<std::uint32_t> draft;
        if (draft_strategy_ && remaining > 1 && !recover_after_rejection) {
            draft = draft_strategy_->propose(
                history,
                static_cast<std::size_t>(std::min(speculative_max_tokens_, remaining - 1))
            );
        }
        recover_after_rejection = false;

        std::vector<std::uint32_t> verification_tokens;
        verification_tokens.reserve(draft.size() + 1U);
        verification_tokens.push_back(previous_token);
        verification_tokens.insert(verification_tokens.end(), draft.begin(), draft.end());
        protocol::StageRequest stage_request = initial_request(
            request,
            "decode",
            decode_step,
            verification_tokens
        );
        PassResult pass = run_pass(request, invoke_stage_, std::move(stage_request), result);
        ++result.target_decode_passes;
        if (!pass.ok) {
            const std::string cleanup_error = close_sessions(request, invoke_stage_);
            if (!cleanup_error.empty()) pass.error.error_message += "; cleanup: " + cleanup_error;
            return pass.error;
        }

        SampledBatch target;
        try {
            target = read_sampled_batch(pass.final_response);
        } catch (const std::exception& error) {
            return fail("502 Bad Gateway", "invalid_sampled_token", error.what());
        }
        if (target.tokens.size() != verification_tokens.size()) {
            return fail(
                "502 Bad Gateway",
                "invalid_speculative_verification",
                "target output count does not match speculative verification input"
            );
        }

        std::size_t accepted = 0;
        while (accepted < draft.size() &&
               !target.end_of_generation[accepted] &&
               target.tokens[accepted] == draft[accepted]) {
            ++accepted;
        }
        result.speculative_draft_tokens += static_cast<int>(draft.size());
        result.speculative_accepted_tokens += static_cast<int>(accepted);

        const int rollback_tokens = accepted < draft.size()
            ? static_cast<int>(draft.size() - accepted)
            : 0;
        if (rollback_tokens > 0) {
            const std::string rollback_error = rollback_sessions(
                request,
                invoke_stage_,
                rollback_tokens,
                result
            );
            if (!rollback_error.empty()) {
                return fail(
                    "502 Bad Gateway",
                    "generation_rollback_failed",
                    rollback_error
                );
            }
            recover_after_rejection = true;
        }
        decode_step += static_cast<int>(verification_tokens.size()) - rollback_tokens;

        const std::size_t output_count = accepted < draft.size()
            ? accepted + 1U
            : target.tokens.size();
        for (std::size_t index = 0;
             index < output_count && result.completion_tokens < request.max_tokens;
             ++index) {
            if (target.end_of_generation[index]) {
                end_of_generation = true;
                break;
            }
            if (auto error = emit(target.tokens[index], target.text[index]); error.has_value()) {
                return *error;
            }
        }
    }

    const std::string cleanup_error = close_sessions(request, invoke_stage_);
    if (!cleanup_error.empty()) {
        return generation_error("502 Bad Gateway", "generation_cleanup_failed", cleanup_error);
    }
    result.ok = true;
    result.status = "200 OK";
    result.finish_reason = end_of_generation ? "stop" : "length";
    return result;
}

} // namespace jetsonfabric::runtime::pipeline_parallel
