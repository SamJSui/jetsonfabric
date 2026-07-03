#pragma once

#include "engine/engine.hpp"

#include <string>

namespace jetsonfabric::runtime {

class StubEngine final : public Engine {
public:
    StubEngine(std::string model, std::string mode);

    std::string runtime_name() const override;
    std::string mode() const override;
    std::string model() const override;
    std::string chat_completion(const std::string& request_body) const override;

private:
    std::string model_;
    std::string mode_;
};

} // namespace jetsonfabric::runtime
