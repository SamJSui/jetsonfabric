#pragma once

#include "deployment/model_manager.hpp"
#include "engine/engine.hpp"
#include "worker/config.hpp"

namespace jetsonfabric::runtime {

// RuntimeService exposes the runtime HTTP contract while ModelManager owns the
// single resident deployment and its execution components.
class RuntimeService final : public RuntimeAPI {
public:
    explicit RuntimeService(Config config);

    std::string runtime_name() const override;
    std::string engine_name() const override;
    ExecutionMode execution_mode() const override;
    std::string model() const override;

    RuntimeResponse deployment_status() const override;
    RuntimeResponse load_deployment(const std::string& request_body) override;
    RuntimeResponse activate_deployment(const std::string& request_body) override;
    RuntimeResponse unload_deployment(const std::string& request_body) override;
    RuntimeResponse chat_completion(const std::string& request_body) const override;
    RuntimeResponse generate(
        const std::string& request_body,
        const GenerationEventSink& sink
    ) const override;
    RuntimeResponse run_stage(const std::string& request_body) const override;

private:
    Config config_;
    deployment::ModelManager model_manager_;
};

} // namespace jetsonfabric::runtime
