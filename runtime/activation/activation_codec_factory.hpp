#pragma once

#include "activation/activation_codec.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace jetsonfabric::runtime::activation {

class ActivationCodecFactory {
public:
    using Builder = std::function<std::shared_ptr<const ActivationCodec>()>;

    void register_codec(std::string name, Builder builder);
    bool supports(const std::string& name) const;
    std::shared_ptr<const ActivationCodec> create_codec(const std::string& name) const;

private:
    std::map<std::string, Builder> builders_;
};

std::shared_ptr<const ActivationCodecFactory> make_default_activation_codec_factory();

} // namespace jetsonfabric::runtime::activation
