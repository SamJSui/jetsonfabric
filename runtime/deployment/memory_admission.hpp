#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace jetsonfabric::runtime::deployment {

constexpr std::uint64_t kMinimumPostLoadHeadroomBytes = 256ULL << 20;

struct LoadMemoryEstimate {
    std::uint64_t resident_weight_bytes = 0;
};

struct MemoryAdmissionDecision {
    bool admitted = true;
    std::optional<std::uint64_t> available_bytes;
    std::optional<LoadMemoryEstimate> estimate;
    std::uint64_t required_bytes = 0;

    std::string rejection_message() const;
};

std::optional<std::uint64_t> available_memory_bytes();

MemoryAdmissionDecision assess_load_memory(
    std::optional<std::uint64_t> available_bytes,
    std::optional<LoadMemoryEstimate> estimate
);

} // namespace jetsonfabric::runtime::deployment
