#include "transport/stage_transport_factory.hpp"

#include "protocol/stage.hpp"
#include "transport/http_stage_transport.hpp"

#include <memory>

namespace jetsonfabric::runtime::transport {

std::shared_ptr<const StageTransportFactory> make_default_stage_transport_factory() {
    auto factory = std::make_shared<StageTransportFactory>();
    factory->register_transport(
        protocol::kStageWireTransport,
        [](const Config& config) {
            return std::make_shared<HTTPStageTransport>(config.cluster_token);
        }
    );
    factory->register_transport(
        protocol::kDirectStageTransport,
        [](const Config& config) {
            return std::make_shared<HTTPStageTransport>(
                config.cluster_token,
                HTTPStageTarget::Runtime
            );
        }
    );
    return factory;
}

} // namespace jetsonfabric::runtime::transport
