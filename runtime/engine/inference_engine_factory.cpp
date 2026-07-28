#include "engine/inference_engine_factory.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime {

void InferenceEngineFactory::register_engine(std::string engine_name, Builder builder) {
    if (engine_name.empty()) {
        throw std::invalid_argument("inference engine name must not be empty");
    }
    if (!builder) {
        throw std::invalid_argument("inference engine builder must not be empty");
    }
    if (!builders_.emplace(engine_name, std::move(builder)).second) {
        throw std::invalid_argument("inference engine is already registered: " + engine_name);
    }
}

bool InferenceEngineFactory::supports(const std::string& engine_name) const {
    return builders_.contains(engine_name);
}

InferenceEngineParts InferenceEngineFactory::create_engine(const Config& config) const {
    const auto builder = builders_.find(config.engine);
    if (builder == builders_.end()) {
        std::ostringstream message;
        message << "unsupported inference engine: " << config.engine;
        if (!builders_.empty()) {
            message << "; registered engines:";
            for (const auto& [name, ignored] : builders_) {
                (void) ignored;
                message << " " << name;
            }
        }
        throw std::invalid_argument(message.str());
    }
    return builder->second(config);
}

} // namespace jetsonfabric::runtime
