#include "speculative/draft_strategy_factory.hpp"

#include "speculative/prompt_lookup_draft.hpp"

#include <stdexcept>

namespace jetsonfabric::runtime::speculative {

std::shared_ptr<const DraftStrategy> create_draft_strategy(const std::string& name) {
    if (name == "none") return nullptr;
    if (name == "prompt_lookup") return std::make_shared<PromptLookupDraft>();
    throw std::invalid_argument("unknown speculative draft strategy: " + name);
}

} // namespace jetsonfabric::runtime::speculative
