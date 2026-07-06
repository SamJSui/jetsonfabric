#pragma once

#include "engine/engine.hpp"
#include "engine/runtime_engine_factory.hpp"
#include "pipeline_parallel/stage_worker.hpp"
#include "worker/config.hpp"

namespace jetsonfabric::runtime {

class RuntimeEngine final : public Engine {
public:
    explicit RuntimeEngine(Config config);

    std::string runtime_name() const override;
    ExecutionMode mode() const override;
    std::string model() const override;

    EngineResponse chat_completion(const std::string& request_body) const override;
    EngineResponse run_stage(const std::string& request_body) const override;

private:
    Config config_;
    RuntimeEngineParts parts_;
    pipeline_parallel::StageWorker stage_worker_;
};

} // namespace jetsonfabric::runtime