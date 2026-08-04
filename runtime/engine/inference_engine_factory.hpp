#pragma once

#include "deployment/deployment.hpp"
#include "deployment/memory_admission.hpp"
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
    using MemoryEstimator =
        std::function<std::optional<deployment::LoadMemoryEstimate>(const Config&)>;

    void register_engine(
        std::string engine_name,
        Builder builder,
        MemoryEstimator memory_estimator = {}
    );
    bool supports(const std::string& engine_name) const;
    InferenceEngineParts create_engine(const Config& config) const;
    std::optional<deployment::LoadMemoryEstimate> estimate_load_memory(
        const Config& config
    ) const;

private:
    struct Registration {
        Builder builder;
        MemoryEstimator memory_estimator;
    };

    std::map<std::string, Registration> registrations_;
};

std::shared_ptr<const InferenceEngineFactory> make_default_inference_engine_factory();

} // namespace jetsonfabric::runtime
