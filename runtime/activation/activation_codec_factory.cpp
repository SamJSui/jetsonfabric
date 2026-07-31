#include "activation/activation_codec_factory.hpp"

#include <stdexcept>
#include <utility>

namespace jetsonfabric::runtime::activation {

void ActivationCodecFactory::register_codec(std::string name, Builder builder) {
    if (name.empty()) {
        throw std::invalid_argument("activation codec name must not be empty");
    }
    if (!builder) {
        throw std::invalid_argument("activation codec builder must not be empty");
    }
    if (!builders_.emplace(std::move(name), std::move(builder)).second) {
        throw std::invalid_argument("activation codec is already registered");
    }
}

bool ActivationCodecFactory::supports(const std::string& name) const {
    return builders_.find(name) != builders_.end();
}

std::shared_ptr<const ActivationCodec> ActivationCodecFactory::create_codec(
    const std::string& name
) const {
    const auto builder = builders_.find(name);
    if (builder == builders_.end()) {
        throw std::invalid_argument("unsupported activation encoding: " + name);
    }
    std::shared_ptr<const ActivationCodec> codec = builder->second();
    if (!codec) {
        throw std::runtime_error("activation codec builder returned null: " + name);
    }
    if (codec->name() != name) {
        throw std::runtime_error(
            "activation codec registered as " + name +
            " returned strategy " + codec->name()
        );
    }
    return codec;
}

} // namespace jetsonfabric::runtime::activation
