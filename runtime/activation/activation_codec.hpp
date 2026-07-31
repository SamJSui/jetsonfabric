#pragma once

#include "inference/stage.hpp"

#include <string>

namespace jetsonfabric::runtime::activation {

// ActivationCodec converts activation tensors between the engine's f32
// representation and the representation carried between pipeline stages.
class ActivationCodec {
public:
    virtual ~ActivationCodec() = default;

    virtual const std::string& name() const noexcept = 0;
    virtual inference::Payload encode(inference::Payload payload) const = 0;
    virtual inference::Payload decode(inference::Payload payload) const = 0;
};

} // namespace jetsonfabric::runtime::activation
