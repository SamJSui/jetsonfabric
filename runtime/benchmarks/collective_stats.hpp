#pragma once

#include <cstddef>
#include <vector>

namespace jetsonfabric::benchmarks {

struct LatencySummary {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
};

LatencySummary summarize_latencies(std::vector<double> latencies_us);

double projected_communication_ms(
    double collective_latency_us,
    std::size_t layer_count,
    std::size_t collectives_per_layer);

}  // namespace jetsonfabric::benchmarks
