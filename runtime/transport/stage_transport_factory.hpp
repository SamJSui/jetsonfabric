#pragma once

#include "transport/stage_transport.hpp"
#include "worker/config.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace jetsonfabric::runtime::transport {

// Constructs the configured peer-stage transport. Transport implementations
// register builders here without changing generation orchestration.
class StageTransportFactory {
public:
    using Builder = std::function<std::shared_ptr<const StageTransport>(const Config&)>;

    void register_transport(std::string transport_name, Builder builder);
    bool supports(const std::string& transport_name) const;
    std::shared_ptr<const StageTransport> create_transport(const Config& config) const;

private:
    std::map<std::string, Builder> builders_;
};

std::shared_ptr<const StageTransportFactory> make_default_stage_transport_factory();

} // namespace jetsonfabric::runtime::transport
