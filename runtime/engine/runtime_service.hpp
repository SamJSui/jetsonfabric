#pragma once

#include "engine/engine.hpp"
#include "engine/inference_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

namespace jetsonfabric::runtime {

// RuntimeService is the JetsonFabric-owned host service. It owns the configured
// inference-engine adapter and exposes the runtime HTTP contract.
class RuntimeService final : public RuntimeAPI {
public:
    explicit RuntimeService(Config config);

    std::string runtime_name() const override;
    std::string engine_name() const override;
    ExecutionMode execution_mode() const override;
    std::string model() const override;

    RuntimeResponse chat_completion(const std::string& request_body) const override;
    RuntimeResponse run_stage(const std::string& request_body) const override;

private:
    Config config_;
    InferenceEngineParts engine_parts_;
    pipeline_parallel::StageWorker stage_worker_;
};

} // namespace jetsonfabric::runtime
