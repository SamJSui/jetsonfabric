#pragma once

#include "speculative/draft_strategy.hpp"

#include <cstddef>

namespace jetsonfabric::runtime::speculative {

class PromptLookupDraft final : public DraftStrategy {
public:
    explicit PromptLookupDraft(std::size_t max_ngram = 4, std::size_t min_ngram = 2);

    std::vector<std::uint32_t> propose(
        const std::vector<std::uint32_t>& history,
        std::size_t max_tokens
    ) const override;

private:
    std::size_t max_ngram_;
    std::size_t min_ngram_;
};

} // namespace jetsonfabric::runtime::speculative
