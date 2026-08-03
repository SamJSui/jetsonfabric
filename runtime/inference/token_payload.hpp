#pragma once

#include "inference/stage.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace jetsonfabric::runtime::inference {

std::vector<std::int32_t> decode_token_ids(const Payload& payload);
std::int32_t decode_single_token(const Payload& payload);
Payload sampled_token_payload(std::int32_t token);
Payload sampled_tokens_payload(std::span<const std::int32_t> tokens);

} // namespace jetsonfabric::runtime::inference
