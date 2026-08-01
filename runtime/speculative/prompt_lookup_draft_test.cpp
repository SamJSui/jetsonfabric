#include "speculative/prompt_lookup_draft.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    using jetsonfabric::runtime::speculative::PromptLookupDraft;

    PromptLookupDraft draft;
    expect(
        draft.propose({1, 2, 3, 1, 2, 3, 1, 2}, 3) == std::vector<std::uint32_t>({3, 1, 2}),
        "prompt lookup did not continue the latest matching suffix"
    );
    expect(draft.propose({1, 2, 3, 4}, 3).empty(), "unique history produced a draft");
    expect(draft.propose({1, 2, 1, 2}, 0).empty(), "zero draft budget produced tokens");

    bool rejected = false;
    try {
        (void) PromptLookupDraft(1, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "invalid ngram bounds were accepted");

    std::cout << "prompt lookup draft strategy passed\n";
    return 0;
}
