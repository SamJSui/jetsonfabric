#include "deployment/memory_admission.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace jetsonfabric::runtime::deployment {
namespace {

constexpr std::uint64_t kBytesPerKiB = 1024;

std::string format_mib(std::uint64_t bytes) {
    std::ostringstream output;
    output << bytes / (1024ULL * 1024ULL) << " MiB";
    return output.str();
}

} // namespace

std::optional<std::uint64_t> available_memory_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t value = 0;
        std::string unit;
        if (!(fields >> key >> value >> unit)) continue;
        if (key != "MemAvailable:") continue;
        if (unit != "kB" || value > std::numeric_limits<std::uint64_t>::max() / kBytesPerKiB) {
            return std::nullopt;
        }
        return value * kBytesPerKiB;
    }
    return std::nullopt;
}

MemoryAdmissionDecision assess_load_memory(
    std::optional<std::uint64_t> available_bytes,
    std::optional<LoadMemoryEstimate> estimate
) {
    MemoryAdmissionDecision decision{
        .available_bytes = available_bytes,
        .estimate = estimate,
    };
    if (!available_bytes.has_value() || !estimate.has_value() ||
        estimate->resident_weight_bytes == 0) {
        return decision;
    }
    if (estimate->resident_weight_bytes >
        std::numeric_limits<std::uint64_t>::max() - kMinimumPostLoadHeadroomBytes) {
        decision.admitted = false;
        decision.required_bytes = std::numeric_limits<std::uint64_t>::max();
        return decision;
    }
    decision.required_bytes =
        estimate->resident_weight_bytes + kMinimumPostLoadHeadroomBytes;
    decision.admitted = *available_bytes >= decision.required_bytes;
    return decision;
}

std::string MemoryAdmissionDecision::rejection_message() const {
    if (admitted || !available_bytes.has_value() || !estimate.has_value()) {
        return {};
    }
    return "deployment needs " + format_mib(estimate->resident_weight_bytes) +
        " of resident weights plus " + format_mib(kMinimumPostLoadHeadroomBytes) +
        " of post-load headroom, but only " + format_mib(*available_bytes) +
        " is available; unload the active deployment or use a smaller stage";
}

} // namespace jetsonfabric::runtime::deployment
