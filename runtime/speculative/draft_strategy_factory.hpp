#pragma once

#include "speculative/draft_strategy.hpp"

#include <memory>
#include <string>

namespace jetsonfabric::runtime::speculative {

std::shared_ptr<const DraftStrategy> create_draft_strategy(const std::string& name);

} // namespace jetsonfabric::runtime::speculative
