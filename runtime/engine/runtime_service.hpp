#pragma once

#include "deployment/model_manager.hpp"
#include "engine/engine.hpp"
#include "worker/config.hpp"

namespace jetsonfabric::runtime {

// RuntimeService is the JetsonFabric-owned host service. It exposes the runtime
// HTTP contract while ModelManager owns the active model execution components.
class RuntimeService final : public RuntimeAPI {
public:
    explicit RuntimeService(Config config);

    std::string runtime_name() const override;
    std::string engine_name() const override;
    ExecutionMode execution_mode() const override;
    std::string model() const override;

    RuntimeResponse deployment_status() const override;
    RuntimeResponse unload_deployment(const std::string& request_body) override;
    RuntimeResponse chat_completion(const std::string& request_body) const override;
    RuntimeResponse run_stage(const std::string& request_body) const override;

private:
    Config config_;
    deployment::ModelManager model_manager_;
};

} // namespace jetsonfabric::runtime