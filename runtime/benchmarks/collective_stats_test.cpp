#include "benchmarks/collective_stats.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

bool close_to(double actual, double expected) {
    return std::abs(actual - expected) < 0.0001;
}

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("collective statistics assertion failed");
    }
}

}  // namespace

int main() {
    using jetsonfabric::benchmarks::projected_communication_ms;
    using jetsonfabric::benchmarks::summarize_latencies;

    const auto summary = summarize_latencies({50.0, 10.0, 40.0, 20.0, 30.0});
    require(close_to(summary.mean_us, 30.0));
    require(close_to(summary.p50_us, 30.0));
    require(close_to(summary.p95_us, 40.0));
    require(close_to(summary.p99_us, 40.0));
    require(close_to(projected_communication_ms(750.0, 48, 2), 72.0));

    bool rejected_empty = false;
    try {
        (void)summarize_latencies({});
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    require(rejected_empty);
}
