#pragma once

#include "deployment/deployment.hpp"
#include "inference/executor.hpp"
#include "pipeline_parallel/stage_assignment.hpp"
#include "worker/config.hpp"

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace jetsonfabric::runtime {

struct InferenceEngineParts {
    std::unique_ptr<inference::Executor> executor;
    std::optional<deployment::ModelResidency> model_residency;
    std::optional<pipeline_parallel::StageAssignment> execution_assignment = std::nullopt;
};

// Dispatches engine construction by configured engine name. Adding an engine
// registers one builder instead of changing runtime orchestration.
class InferenceEngineFactory {
public:
    using Builder = std::function<InferenceEngineParts(const Config&)>;

    void register_engine(std::string engine_name, Builder builder);
    bool supports(const std::string& engine_name) const;
    InferenceEngineParts create_engine(const Config& config) const;

private:
    std::map<std::string, Builder> builders_;
};

std::shared_ptr<const InferenceEngineFactory> make_default_inference_engine_factory();

} // namespace jetsonfabric::runtime
