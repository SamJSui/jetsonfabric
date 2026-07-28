#include "engine/inference_engine_factory.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

template <typename Callable>
void expect_invalid_argument(Callable callable, const std::string& expected_text) {
    try {
        callable();
        expect(false, "expected invalid_argument");
    } catch (const std::invalid_argument& error) {
        expect(
            std::string(error.what()).find(expected_text) != std::string::npos,
            "unexpected invalid_argument: " + std::string(error.what())
        );
    }
}

} // namespace

int main() {
    using jetsonfabric::runtime::Config;
    using jetsonfabric::runtime::InferenceEngineFactory;
    using jetsonfabric::runtime::InferenceEngineParts;

    InferenceEngineFactory factory;
    int builds = 0;
    factory.register_engine("recording", [&builds](const Config& config) {
        expect(config.model == "test-model", "factory did not receive deployment config");
        ++builds;
        return InferenceEngineParts{};
    });

    expect(factory.supports("recording"), "registered engine was not discoverable");
    expect(!factory.supports("missing"), "unknown engine was reported as supported");

    Config config;
    config.engine = "recording";
    config.model = "test-model";
    (void) factory.create_engine(config);
    expect(builds == 1, "registered engine builder was not called exactly once");

    expect_invalid_argument(
        [&factory]() { factory.register_engine("", [](const Config&) {
            return InferenceEngineParts{};
        }); },
        "name must not be empty"
    );
    expect_invalid_argument(
        [&factory]() { factory.register_engine("recording", [](const Config&) {
            return InferenceEngineParts{};
        }); },
        "already registered"
    );

    config.engine = "missing";
    expect_invalid_argument(
        [&factory, &config]() { (void) factory.create_engine(config); },
        "registered engines: recording"
    );

    std::cout << "inference engine factory tests passed\n";
    return 0;
}
