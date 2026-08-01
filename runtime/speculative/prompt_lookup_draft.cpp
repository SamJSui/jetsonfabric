#include "speculative/prompt_lookup_draft.hpp"

#include <algorithm>
#include <stdexcept>

namespace jetsonfabric::runtime::speculative {

PromptLookupDraft::PromptLookupDraft(std::size_t max_ngram, std::size_t min_ngram)
    : max_ngram_(max_ngram), min_ngram_(min_ngram) {
    if (min_ngram == 0 || max_ngram < min_ngram) {
        throw std::invalid_argument("prompt lookup requires 0 < min_ngram <= max_ngram");
    }
}

std::vector<std::uint32_t> PromptLookupDraft::propose(
    const std::vector<std::uint32_t>& history,
    std::size_t max_tokens
) const {
    if (max_tokens == 0 || history.size() <= min_ngram_) {
        return {};
    }

    const std::size_t largest_ngram = std::min(max_ngram_, history.size() - 1U);
    for (std::size_t ngram = largest_ngram; ngram >= min_ngram_; --ngram) {
        const std::size_t suffix_start = history.size() - ngram;
        for (std::size_t candidate = suffix_start; candidate-- > 0U;) {
            if (candidate + ngram > suffix_start ||
                !std::equal(
                    history.begin() + static_cast<std::ptrdiff_t>(candidate),
                    history.begin() + static_cast<std::ptrdiff_t>(candidate + ngram),
                    history.begin() + static_cast<std::ptrdiff_t>(suffix_start))) {
                continue;
            }
            const std::size_t proposal_start = candidate + ngram;
            const std::size_t proposal_count = std::min(max_tokens, history.size() - proposal_start);
            return std::vector<std::uint32_t>(
                history.begin() + static_cast<std::ptrdiff_t>(proposal_start),
                history.begin() + static_cast<std::ptrdiff_t>(proposal_start + proposal_count)
            );
        }
        if (ngram == min_ngram_) break;
    }
    return {};
}

} // namespace jetsonfabric::runtime::speculative
