#include "transport/stage_transport_factory.hpp"

#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::transport {

void StageTransportFactory::register_transport(
    std::string transport_name,
    Builder builder
) {
    if (transport_name.empty()) {
        throw std::invalid_argument("stage transport name is required");
    }
    if (!builder) {
        throw std::invalid_argument("stage transport builder is required");
    }
    if (!builders_.emplace(std::move(transport_name), std::move(builder)).second) {
        throw std::invalid_argument("stage transport is already registered");
    }
}

bool StageTransportFactory::supports(const std::string& transport_name) const {
    return builders_.contains(transport_name);
}

std::shared_ptr<const StageTransport> StageTransportFactory::create_transport(
    const Config& config
) const {
    const auto builder = builders_.find(config.stage_transport);
    if (builder == builders_.end()) {
        throw std::invalid_argument(
            "unsupported stage transport: " + config.stage_transport
        );
    }
    std::shared_ptr<const StageTransport> transport = builder->second(config);
    if (!transport) {
        throw std::invalid_argument(
            "stage transport builder returned no transport: " + config.stage_transport
        );
    }
    return transport;
}

} // namespace jetsonfabric::runtime::transport
