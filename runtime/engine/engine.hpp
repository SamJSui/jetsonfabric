#pragma once

#include <string>

namespace jetsonfabric::runtime {

class Engine {
public:
    virtual ~Engine() = default;

    virtual std::string runtime_name() const = 0;
    virtual std::string mode() const = 0;
    virtual std::string model() const = 0;
    virtual std::string chat_completion(const std::string& request_body) const = 0;
};

} // namespace jetsonfabric::runtime
