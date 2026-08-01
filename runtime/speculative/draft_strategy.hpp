#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jetsonfabric::runtime::speculative {

class DraftStrategy {
public:
    virtual ~DraftStrategy() = default;

    virtual std::vector<std::uint32_t> propose(
        const std::vector<std::uint32_t>& history,
        std::size_t max_tokens
    ) const = 0;
};

} // namespace jetsonfabric::runtime::speculative
