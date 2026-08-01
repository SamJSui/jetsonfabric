#include "jetsonfabric/native_inference.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string package_path;
    jetsonfabric::native::Backend backend = jetsonfabric::native::Backend::Cpu;
    std::vector<std::int32_t> tokens;
    std::uint32_t max_tokens = 1;
    std::uint32_t warmups = 0;
    std::uint32_t iterations = 1;
    int threads = 1;
    std::optional<std::int32_t> expected_first_token;
};

std::uint32_t parse_u32(const std::string& value, const char * name) {
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed > UINT32_MAX) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::vector<std::int32_t> parse_tokens(const std::string& value) {
    std::vector<std::int32_t> tokens;
    std::size_t start = 0;
    while (start < value.size()) {
        const std::size_t end = value.find(',', start);
        const std::string part = value.substr(start, end - start);
        std::size_t consumed = 0;
        const long parsed = std::stol(part, &consumed);
        if (consumed != part.size() || parsed < 0 || parsed > INT32_MAX) {
            throw std::invalid_argument("invalid --tokens value");
        }
        tokens.push_back(static_cast<std::int32_t>(parsed));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (tokens.empty()) throw std::invalid_argument("--tokens cannot be empty");
    return tokens;
}

Arguments parse_args(int argc, char ** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--package") args.package_path = value;
        else if (option == "--tokens") args.tokens = parse_tokens(value);
        else if (option == "--max-tokens") args.max_tokens = parse_u32(value, "--max-tokens");
        else if (option == "--warmups") args.warmups = parse_u32(value, "--warmups");
        else if (option == "--iterations") args.iterations = parse_u32(value, "--iterations");
        else if (option == "--threads") args.threads = static_cast<int>(parse_u32(value, "--threads"));
        else if (option == "--expected-first-token") {
            args.expected_first_token = static_cast<std::int32_t>(parse_u32(value, option.c_str()));
        } else if (option == "--backend" && value == "cpu") args.backend = jetsonfabric::native::Backend::Cpu;
        else if (option == "--backend" && value == "cuda") args.backend = jetsonfabric::native::Backend::Cuda;
        else throw std::invalid_argument("unknown option or backend: " + option + " " + value);
    }
    if (args.package_path.empty() || args.tokens.empty() || args.max_tokens == 0 ||
        args.iterations == 0 || args.threads <= 0) {
        throw std::invalid_argument(
            "--package, --tokens, positive --max-tokens, --iterations, and --threads are required"
        );
    }
    return args;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}

void print_tokens(const std::vector<std::int32_t>& tokens) {
    std::cout << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        jetsonfabric::native::NativeEngine engine(args.package_path, args.backend, args.threads);
        for (std::uint32_t index = 0; index < args.warmups; ++index) {
            (void) engine.generate(args.tokens, args.max_tokens);
        }

        std::vector<double> ttft;
        std::vector<double> itl;
        std::vector<double> throughput;
        std::vector<std::int32_t> sampled_tokens;
        for (std::uint32_t index = 0; index < args.iterations; ++index) {
            const auto result = engine.generate(args.tokens, args.max_tokens);
            if (index == 0) sampled_tokens = result.sampled_tokens;
            if (result.sampled_tokens != sampled_tokens) {
                throw std::runtime_error("native greedy output changed between iterations");
            }
            ttft.push_back(result.time_to_first_token_ms);
            itl.push_back(result.inter_token_latency_ms);
            throughput.push_back(result.output_tokens_per_second);
        }
        if (args.expected_first_token && sampled_tokens.front() != *args.expected_first_token) {
            throw std::runtime_error("native first token does not match expected oracle token");
        }

        const auto& info = engine.model_info();
        std::cout << std::fixed << std::setprecision(3)
                  << "{\n"
                  << "  \"engine\": \"jetsonfabric-native\",\n"
                  << "  \"architecture\": \"" << info.architecture << "\",\n"
                  << "  \"backend\": \"" << jetsonfabric::native::backend_name(args.backend) << "\",\n"
                  << "  \"layer_count\": " << info.layer_count << ",\n"
                  << "  \"embedding_length\": " << info.embedding_length << ",\n"
                  << "  \"vocabulary_size\": " << info.vocabulary_size << ",\n"
                  << "  \"weight_bytes\": " << info.weight_bytes << ",\n"
                  << "  \"prompt_tokens\": ";
        print_tokens(args.tokens);
        std::cout << ",\n  \"sampled_tokens\": ";
        print_tokens(sampled_tokens);
        std::cout << ",\n"
                  << "  \"iterations\": " << args.iterations << ",\n"
                  << "  \"load_ms\": " << engine.load_time_ms() << ",\n"
                  << "  \"ttft_p50_ms\": " << median(ttft) << ",\n"
                  << "  \"itl_p50_ms\": " << median(itl) << ",\n"
                  << "  \"output_tokens_per_second_p50\": " << median(throughput) << ",\n"
                  << "  \"kv_cache\": false,\n"
                  << "  \"decode_policy\": \"full_prefix_recompute\"\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-native-inference-bench: " << error.what() << '\n';
        return 1;
    }
}
