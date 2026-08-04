#include "engine/inference_engine_factory.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime {

void InferenceEngineFactory::register_engine(
    std::string engine_name,
    Builder builder,
    MemoryAdmissionPolicy memory_admission_policy,
    MemoryEstimator memory_estimator
) {
    if (engine_name.empty()) {
        throw std::invalid_argument("inference engine name must not be empty");
    }
    if (!builder) {
        throw std::invalid_argument("inference engine builder must not be empty");
    }
    if (memory_admission_policy == MemoryAdmissionPolicy::EstimateRequired &&
        !memory_estimator) {
        throw std::invalid_argument(
            "estimated inference engine admission requires a memory estimator"
        );
    }
    if (memory_admission_policy == MemoryAdmissionPolicy::BestEffort &&
        memory_estimator) {
        throw std::invalid_argument(
            "best-effort inference engine admission must not register an estimator"
        );
    }
    if (!registrations_.emplace(
            engine_name,
            Registration{
                .builder = std::move(builder),
                .memory_admission_policy = memory_admission_policy,
                .memory_estimator = std::move(memory_estimator),
            }
        ).second) {
        throw std::invalid_argument("inference engine is already registered: " + engine_name);
    }
}

bool InferenceEngineFactory::supports(const std::string& engine_name) const {
    return registrations_.contains(engine_name);
}

InferenceEngineParts InferenceEngineFactory::create_engine(const Config& config) const {
    const auto registration = registrations_.find(config.engine);
    if (registration == registrations_.end()) {
        std::ostringstream message;
        message << "unsupported inference engine: " << config.engine;
        if (!registrations_.empty()) {
            message << "; registered engines:";
            for (const auto& [name, ignored] : registrations_) {
                (void) ignored;
                message << " " << name;
            }
        }
        throw std::invalid_argument(message.str());
    }
    return registration->second.builder(config);
}

std::optional<deployment::LoadMemoryEstimate>
InferenceEngineFactory::estimate_load_memory(const Config& config) const {
    const auto registration = registrations_.find(config.engine);
    if (registration == registrations_.end()) {
        throw std::invalid_argument("unsupported inference engine: " + config.engine);
    }
    if (registration->second.memory_admission_policy ==
        MemoryAdmissionPolicy::BestEffort) {
        return std::nullopt;
    }
    deployment::LoadMemoryEstimate estimate =
        registration->second.memory_estimator(config);
    if (estimate.resident_weight_bytes == 0) {
        throw std::invalid_argument(
            "estimated inference engine admission returned zero resident weights"
        );
    }
    return estimate;
}

} // namespace jetsonfabric::runtime
