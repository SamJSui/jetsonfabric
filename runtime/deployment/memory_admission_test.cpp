#include "deployment/memory_admission.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

using jetsonfabric::runtime::deployment::LoadMemoryEstimate;
using jetsonfabric::runtime::deployment::assess_load_memory;
using jetsonfabric::runtime::deployment::kMinimumPostLoadHeadroomBytes;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t execution_reserve = 128ULL * 1024ULL * 1024ULL;
    const LoadMemoryEstimate weights{
        .resident_weight_bytes = 4 * gib,
        .reserved_execution_bytes = execution_reserve,
    };

    const auto admitted = assess_load_memory(5 * gib, weights);
    expect(admitted.admitted, "load with sufficient headroom was rejected");
    expect(
        admitted.required_bytes ==
            4 * gib + execution_reserve + kMinimumPostLoadHeadroomBytes,
        "admission calculated the wrong required bytes"
    );

    const auto rejected = assess_load_memory(4 * gib + execution_reserve, weights);
    expect(!rejected.admitted, "load without post-load headroom was admitted");
    expect(
        rejected.rejection_message().find("unload the active deployment") != std::string::npos,
        "rejection did not provide an actionable recovery"
    );

    expect(
        assess_load_memory(std::nullopt, weights).admitted,
        "missing host telemetry should preserve best-effort loading"
    );
    expect(
        assess_load_memory(5 * gib, std::nullopt).admitted,
        "an engine without an estimator should preserve best-effort loading"
    );
    expect(
        assess_load_memory(5 * gib, LoadMemoryEstimate{}).admitted,
        "a zero-byte estimate should not create a false rejection"
    );

    const auto overflow = assess_load_memory(
        std::numeric_limits<std::uint64_t>::max(),
        LoadMemoryEstimate{.resident_weight_bytes = std::numeric_limits<std::uint64_t>::max()}
    );
    expect(!overflow.admitted, "overflowing required bytes were admitted");
    const auto execution_overflow = assess_load_memory(
        std::numeric_limits<std::uint64_t>::max(),
        LoadMemoryEstimate{
            .resident_weight_bytes = 1,
            .reserved_execution_bytes = std::numeric_limits<std::uint64_t>::max(),
        }
    );
    expect(!execution_overflow.admitted, "overflowing execution reserve was admitted");

    const auto observed = jetsonfabric::runtime::deployment::available_memory_bytes();
    expect(!observed.has_value() || *observed > 0, "MemAvailable parser returned zero bytes");

    std::cout << "memory admission tests passed\n";
    return 0;
}
