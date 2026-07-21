#pragma once

#include "pipeline_parallel/stage_result.hpp"
#include "protocol/generation.hpp"
#include "protocol/stage.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::pipeline_parallel {

enum class StageOperation {
    Execute,
    CloseSession,
};

struct GenerationToken {
    std::uint32_t token = 0;
    std::string text;
    int index = 0;
};

struct GenerationResult {
    bool ok = false;
    std::string status = "500 Internal Server Error";
    std::string error_code;
    std::string error_message;
    std::string finish_reason;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::vector<std::uint32_t> sampled_tokens;
    int stage_calls = 0;
    int remote_stage_calls = 0;
    std::int64_t bytes_in = 0;
    std::int64_t bytes_out = 0;
};

using StageInvoker = std::function<StageRunResult(
    const protocol::GenerationStage&,
    const protocol::StageRequest&,
    StageOperation
)>;
using TokenSink = std::function<bool(const GenerationToken&)>;

class GenerationRunner {
public:
    explicit GenerationRunner(StageInvoker invoke_stage);

    GenerationResult run(const protocol::GenerationRequest& request, const TokenSink& sink) const;

private:
    StageInvoker invoke_stage_;
};

} // namespace jetsonfabric::runtime::pipeline_parallel
