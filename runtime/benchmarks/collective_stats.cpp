#include "benchmarks/collective_stats.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace jetsonfabric::benchmarks {
namespace {

double percentile(const std::vector<double>& sorted_values, double fraction) {
    const auto index = static_cast<std::size_t>(
        fraction * static_cast<double>(sorted_values.size() - 1));
    return sorted_values[index];
}

}  // namespace

LatencySummary summarize_latencies(std::vector<double> latencies_us) {
    if (latencies_us.empty()) {
        throw std::invalid_argument("at least one latency sample is required");
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    const double total =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);

    return LatencySummary{
        .mean_us = total / static_cast<double>(latencies_us.size()),
        .p50_us = percentile(latencies_us, 0.50),
        .p95_us = percentile(latencies_us, 0.95),
        .p99_us = percentile(latencies_us, 0.99),
    };
}

double projected_communication_ms(
    double collective_latency_us,
    std::size_t layer_count,
    std::size_t collectives_per_layer) {
    return collective_latency_us * static_cast<double>(layer_count) *
           static_cast<double>(collectives_per_layer) / 1000.0;
}

}  // namespace jetsonfabric::benchmarks
