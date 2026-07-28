#include "transport/stage_transport_factory.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace runtime = jetsonfabric::runtime;

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class TestTransport final : public runtime::transport::StageTransport {
public:
    runtime::pipeline_parallel::StageRunResult invoke(
        const runtime::protocol::GenerationStage&,
        const runtime::protocol::StageRequest&,
        runtime::pipeline_parallel::StageOperation
    ) const override {
        return {};
    }
};

void test_factory_dispatches_registered_transport() {
    runtime::transport::StageTransportFactory factory;
    int builds = 0;
    factory.register_transport("test", [&builds](const runtime::Config&) {
        ++builds;
        return std::make_shared<TestTransport>();
    });
    runtime::Config config;
    config.stage_transport = "test";

    const auto transport = factory.create_transport(config);

    expect(transport != nullptr, "factory returned no transport");
    expect(builds == 1, "factory did not invoke the selected builder");
    expect(factory.supports("test"), "registered transport is not discoverable");
}

void test_factory_rejects_unknown_transport() {
    runtime::transport::StageTransportFactory factory;
    runtime::Config config;
    config.stage_transport = "missing";

    bool rejected = false;
    try {
        (void) factory.create_transport(config);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("missing") != std::string::npos;
    }
    expect(rejected, "factory accepted an unknown transport");
}

} // namespace

int main() {
    try {
        test_factory_dispatches_registered_transport();
        test_factory_rejects_unknown_transport();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "stage transport factory tests passed\n";
    return 0;
}
