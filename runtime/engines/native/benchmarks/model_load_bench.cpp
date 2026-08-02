#include "jetsonfabric/engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments {
    std::string package;
    std::uint32_t layer_start = 0;
    std::uint32_t layer_end = 0;
    int iterations = 5;
    bool verify = false;
    bool prefetch = false;
    bool evict = false;
    std::uint32_t prefetch_threads = 1;
};

struct Summary {
    double minimum_ms = 0;
    double median_ms = 0;
    double p95_ms = 0;
    double maximum_ms = 0;
    bool has_p95 = false;
};

std::uint32_t parse_u32(const std::string& value, const std::string& option) {
    std::size_t parsed = 0;
    const unsigned long result = std::stoul(value, &parsed);
    if (parsed != value.size() || result > UINT32_MAX) {
        throw std::invalid_argument(option + " must be a uint32");
    }
    return static_cast<std::uint32_t>(result);
}

Arguments parse_arguments(int argc, char ** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--verify") {
            arguments.verify = true;
            continue;
        }
        if (option == "--prefetch") {
            arguments.prefetch = true;
            continue;
        }
        if (option == "--evict") {
            arguments.evict = true;
            continue;
        }
        if (option == "--help") {
            std::cout
                << "Usage: jf-native-model-bench --package model.jfm --layer-start N --layer-end N "
                   "[--iterations N] [--verify] [--evict] [--prefetch] [--prefetch-threads N]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--package") {
            arguments.package = value;
        } else if (option == "--layer-start") {
            arguments.layer_start = parse_u32(value, option);
        } else if (option == "--layer-end") {
            arguments.layer_end = parse_u32(value, option);
        } else if (option == "--iterations") {
            arguments.iterations = static_cast<int>(parse_u32(value, option));
        } else if (option == "--prefetch-threads") {
            arguments.prefetch_threads = parse_u32(value, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.package.empty() || arguments.layer_start >= arguments.layer_end) {
        throw std::invalid_argument("package and a non-empty layer range are required");
    }
    if (arguments.iterations <= 0 || arguments.iterations > 10000) {
        throw std::invalid_argument("iterations must be between 1 and 10000");
    }
    if (arguments.prefetch_threads == 0 || arguments.prefetch_threads > 256) {
        throw std::invalid_argument("prefetch threads must be between 1 and 256");
    }
    if (arguments.verify && arguments.prefetch) {
        throw std::invalid_argument(
            "--verify and --prefetch must be measured separately because verification faults pages"
        );
    }
    return arguments;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Summary summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](double fraction) {
        const std::size_t rank = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(samples.size()))
        );
        const std::size_t index = std::max<std::size_t>(1, rank) - 1;
        return samples[index];
    };
    return Summary{
        .minimum_ms = samples.front(),
        .median_ms = percentile(0.5),
        .p95_ms = percentile(0.95),
        .maximum_ms = samples.back(),
        .has_p95 = samples.size() >= 20,
    };
}

void print_summary(std::string_view name, Summary summary, bool trailing_comma) {
    std::cout
        << "  \"" << name << "\": {"
        << "\"min_ms\": " << summary.minimum_ms << ", "
        << "\"median_ms\": " << summary.median_ms << ", "
        << "\"max_ms\": " << summary.maximum_ms;
    if (summary.has_p95) {
        std::cout << ", \"p95_ms\": " << summary.p95_ms;
    }
    std::cout << "}"
        << (trailing_comma ? "," : "") << "\n";
}

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(digest.size() * 2);
    for (const std::uint8_t byte : digest) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

void run(const Arguments& arguments) {
    const jf_stage_plan plan = {
        .layer_start = arguments.layer_start,
        .layer_end = arguments.layer_end,
        .verify_hashes = arguments.verify ? 1U : 0U,
        .evict_before_open = arguments.evict ? 1U : 0U,
    };
    std::vector<double> open_samples;
    std::vector<double> prefetch_samples;
    std::vector<double> readiness_samples;
    jf_model_stats stats{};
    std::array<std::uint8_t, 32> source_sha256{};
    std::uint64_t checksum = 0;

    for (int iteration = 0; iteration < arguments.iterations; ++iteration) {
        jf_model * model = nullptr;
        const Clock::time_point open_start = Clock::now();
        const jf_status open_status = jf_model_open(arguments.package.c_str(), &plan, &model);
        const Clock::time_point open_end = Clock::now();
        if (open_status.code != JF_STATUS_OK) {
            throw std::runtime_error(
                std::string(jf_status_code_name(open_status.code)) + ": " + open_status.message
            );
        }
        open_samples.push_back(elapsed_ms(open_start, open_end));
        stats = jf_model_get_stats(model);
        const jf_status identity_status = jf_model_get_source_sha256(
            model,
            source_sha256.data()
        );
        if (identity_status.code != JF_STATUS_OK) {
            jf_model_close(model);
            throw std::runtime_error(identity_status.message);
        }

        if (arguments.prefetch) {
            const Clock::time_point prefetch_start = Clock::now();
            const jf_status prefetch_status = jf_model_prefetch_parallel(
                model,
                arguments.prefetch_threads,
                &checksum
            );
            const Clock::time_point prefetch_end = Clock::now();
            if (prefetch_status.code != JF_STATUS_OK) {
                jf_model_close(model);
                throw std::runtime_error(prefetch_status.message);
            }
            prefetch_samples.push_back(elapsed_ms(prefetch_start, prefetch_end));
            readiness_samples.push_back(elapsed_ms(open_start, prefetch_end));
        }
        jf_model_close(model);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << "{\n"
        << "  \"benchmark\": \"jf_native_model_load_v2\",\n"
        << "  \"iterations\": " << arguments.iterations << ",\n"
        << "  \"verify_hashes\": " << (arguments.verify ? "true" : "false") << ",\n"
        << "  \"evict_before_open\": " << (arguments.evict ? "true" : "false") << ",\n"
        << "  \"prefetch\": " << (arguments.prefetch ? "true" : "false") << ",\n"
        << "  \"prefetch_threads\": " << arguments.prefetch_threads << ",\n"
        << "  \"layer_start\": " << stats.layer_start << ",\n"
        << "  \"layer_end\": " << stats.layer_end << ",\n"
        << "  \"layer_count\": " << stats.layer_count << ",\n"
        << "  \"source_sha256\": \"" << hex_digest(source_sha256) << "\",\n"
        << "  \"selected_weight_bytes\": " << stats.selected_weight_bytes << ",\n"
        << "  \"total_weight_bytes\": " << stats.total_weight_bytes << ",\n"
        << "  \"mapped_bytes\": " << stats.mapped_bytes << ",\n"
        << "  \"selected_tensor_count\": " << stats.selected_tensor_count << ",\n"
        << "  \"total_tensor_count\": " << stats.total_tensor_count << ",\n"
        << "  \"prefetch_checksum\": " << checksum << ",\n";
    print_summary("open", summarize(open_samples), arguments.prefetch);
    if (arguments.prefetch) {
        print_summary("prefetch", summarize(prefetch_samples), true);
        print_summary("ready", summarize(readiness_samples), false);
    }
    std::cout << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        run(parse_arguments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-native-model-bench: " << error.what() << '\n';
        return 1;
    }
}
