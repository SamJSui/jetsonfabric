#include "jetsonfabric/native_inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class SessionPolicy { Warm, Cold, Mixed };

struct Arguments {
    std::string package_path;
    jetsonfabric::native::Backend backend = jetsonfabric::native::Backend::Cpu;
    jetsonfabric::native::AttentionKernel prefill_attention_kernel =
        jetsonfabric::native::AttentionKernel::Automatic;
    jetsonfabric::native::AttentionKernel decode_attention_kernel =
        jetsonfabric::native::AttentionKernel::Automatic;
    std::vector<std::int32_t> tokens;
    std::uint32_t max_tokens = 1;
    std::uint32_t warmups = 0;
    std::uint32_t iterations = 1;
    int threads = 1;
    bool incremental = true;
    SessionPolicy session_policy = SessionPolicy::Warm;
    std::optional<std::vector<std::int32_t>> alternate_tokens;
    std::optional<std::vector<std::int32_t>> expected_tokens;
    std::optional<std::vector<std::int32_t>> expected_alternate_tokens;
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

jetsonfabric::native::AttentionKernel parse_attention_kernel(
    const std::string& value,
    const std::string& option
) {
    if (value == "auto") return jetsonfabric::native::AttentionKernel::Automatic;
    if (value == "unfused") return jetsonfabric::native::AttentionKernel::Unfused;
    if (value == "flash") return jetsonfabric::native::AttentionKernel::Flash;
    throw std::invalid_argument(option + " must be auto, unfused, or flash");
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
        else if (option == "--expected-tokens") args.expected_tokens = parse_tokens(value);
        else if (option == "--expected-alternate-tokens") {
            args.expected_alternate_tokens = parse_tokens(value);
        }
        else if (option == "--alternate-tokens") args.alternate_tokens = parse_tokens(value);
        else if (option == "--session-policy" && value == "warm") {
            args.session_policy = SessionPolicy::Warm;
        } else if (option == "--session-policy" && value == "cold") {
            args.session_policy = SessionPolicy::Cold;
        } else if (option == "--session-policy" && value == "mixed") {
            args.session_policy = SessionPolicy::Mixed;
        }
        else if (option == "--decode-policy" && value == "incremental") args.incremental = true;
        else if (option == "--decode-policy" && value == "full-prefix") args.incremental = false;
        else if (option == "--backend" && value == "cpu") args.backend = jetsonfabric::native::Backend::Cpu;
        else if (option == "--backend" && value == "cuda") args.backend = jetsonfabric::native::Backend::Cuda;
        else if (option == "--prefill-attention-kernel") {
            args.prefill_attention_kernel = parse_attention_kernel(value, option);
        } else if (option == "--decode-attention-kernel") {
            args.decode_attention_kernel = parse_attention_kernel(value, option);
        }
        else throw std::invalid_argument("unknown option or backend: " + option + " " + value);
    }
    if (args.package_path.empty() || args.tokens.empty() || args.max_tokens == 0 ||
        args.iterations == 0 || args.threads <= 0) {
        throw std::invalid_argument(
            "--package, --tokens, positive --max-tokens, --iterations, and --threads are required"
        );
    }
    if (args.session_policy == SessionPolicy::Mixed &&
        (!args.incremental || !args.alternate_tokens)) {
        throw std::invalid_argument(
            "--session-policy mixed requires incremental decode and --alternate-tokens"
        );
    }
    return args;
}

const char * session_policy_name(SessionPolicy policy) {
    switch (policy) {
    case SessionPolicy::Warm: return "exact_shape_reuse_enabled";
    case SessionPolicy::Cold: return "fresh_session";
    case SessionPolicy::Mixed: return "mixed_shape";
    }
    throw std::logic_error("unknown session policy");
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

void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << std::fixed << std::setprecision(3) << samples[index];
    }
    std::cout << ']';
}

std::int32_t greedy_token(const std::vector<float>& logits) {
    if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float value) {
        return std::isfinite(value);
    })) {
        throw std::runtime_error("native benchmark received invalid logits");
    }
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))
    );
}

jetsonfabric::native::GenerationResult generate_full_prefix(
    jetsonfabric::native::NativeEngine& engine,
    const std::vector<std::int32_t>& prompt,
    std::uint32_t max_tokens
) {
    std::vector<std::int32_t> sequence = prompt;
    jetsonfabric::native::GenerationResult result;
    double total_ms = 0.0;
    double decode_ms = 0.0;
    for (std::uint32_t index = 0; index < max_tokens; ++index) {
        const auto start = std::chrono::steady_clock::now();
        const std::int32_t token = greedy_token(engine.logits(sequence));
        const double token_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();
        if (index == 0) result.time_to_first_token_ms = token_ms;
        else decode_ms += token_ms;
        total_ms += token_ms;
        result.sampled_tokens.push_back(token);
        sequence.push_back(token);
    }
    if (max_tokens > 1) {
        result.inter_token_latency_ms = decode_ms / static_cast<double>(max_tokens - 1);
        result.decode_tokens_per_second = 1000.0 / result.inter_token_latency_ms;
    }
    result.end_to_end_tokens_per_second =
        1000.0 * static_cast<double>(max_tokens) / total_ms;
    return result;
}

struct RunResult {
    jetsonfabric::native::GenerationResult primary;
    std::optional<jetsonfabric::native::GenerationResult> alternate;
};

RunResult run_generation(
    jetsonfabric::native::NativeEngine& engine,
    const Arguments& args
) {
    if (args.session_policy == SessionPolicy::Cold) engine.release_session();
    RunResult run;
    run.primary = args.incremental
        ? engine.generate(args.tokens, args.max_tokens)
        : generate_full_prefix(engine, args.tokens, args.max_tokens);
    if (args.session_policy == SessionPolicy::Mixed) {
        const std::size_t capacity = args.tokens.size() + args.max_tokens - 1U;
        if (args.alternate_tokens->size() > capacity) {
            throw std::invalid_argument("alternate tokens exceed the primary session capacity");
        }
        const auto alternate_max_tokens = static_cast<std::uint32_t>(
            capacity - args.alternate_tokens->size() + 1U
        );
        run.alternate = engine.generate(*args.alternate_tokens, alternate_max_tokens);
    }
    return run;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        jetsonfabric::native::NativeEngine engine(
            args.package_path,
            jetsonfabric::native::EngineOptions{
                .backend = args.backend,
                .prefill_attention_kernel = args.prefill_attention_kernel,
                .decode_attention_kernel = args.decode_attention_kernel,
                .threads = args.threads,
            }
        );
        for (std::uint32_t index = 0; index < args.warmups; ++index) {
            (void) run_generation(engine, args);
        }

        std::vector<double> ttft;
        std::vector<double> itl;
        std::vector<double> decode_throughput;
        std::vector<double> end_to_end_throughput;
        std::vector<double> prefill_graph_build;
        std::vector<double> prefill_graph_allocation;
        std::vector<double> prefill_host_input_preparation;
        std::vector<double> prefill_compute;
        std::vector<double> prefill_output_read;
        std::uint32_t prefill_plan_reuse_count = 0;
        std::uint32_t prefill_attention_backend_verified_count = 0;
        jetsonfabric::native::ExecutionBufferMetrics buffers;
        std::vector<std::int32_t> sampled_tokens;
        std::vector<std::int32_t> alternate_sampled_tokens;
        std::vector<double> alternate_ttft;
        std::vector<double> alternate_end_to_end_throughput;
        for (std::uint32_t index = 0; index < args.iterations; ++index) {
            const RunResult run = run_generation(engine, args);
            const auto& result = run.primary;
            if (index == 0) sampled_tokens = result.sampled_tokens;
            if (result.sampled_tokens != sampled_tokens) {
                throw std::runtime_error("native greedy output changed between iterations");
            }
            ttft.push_back(result.time_to_first_token_ms);
            itl.push_back(result.inter_token_latency_ms);
            decode_throughput.push_back(result.decode_tokens_per_second);
            end_to_end_throughput.push_back(result.end_to_end_tokens_per_second);
            prefill_graph_build.push_back(result.prefill.planning_ms);
            prefill_graph_allocation.push_back(result.prefill.allocation_ms);
            prefill_host_input_preparation.push_back(
                result.prefill.host_input_preparation_ms
            );
            prefill_compute.push_back(result.prefill.compute_ms);
            prefill_output_read.push_back(result.prefill.output_read_ms);
            if (result.prefill.plan_reused) ++prefill_plan_reuse_count;
            if (result.prefill.attention_backend_verified) {
                ++prefill_attention_backend_verified_count;
            }
            buffers = result.buffers;
            if (run.alternate) {
                if (index == 0) alternate_sampled_tokens = run.alternate->sampled_tokens;
                if (run.alternate->sampled_tokens != alternate_sampled_tokens) {
                    throw std::runtime_error(
                        "native alternate greedy output changed between iterations"
                    );
                }
                alternate_ttft.push_back(run.alternate->time_to_first_token_ms);
                alternate_end_to_end_throughput.push_back(
                    run.alternate->end_to_end_tokens_per_second
                );
                buffers = run.alternate->buffers;
            }
        }
        if (args.expected_tokens && sampled_tokens != *args.expected_tokens) {
            throw std::runtime_error("native token sequence does not match expected oracle tokens");
        }
        if (args.expected_alternate_tokens &&
            alternate_sampled_tokens != *args.expected_alternate_tokens) {
            throw std::runtime_error(
                "native alternate token sequence does not match expected oracle tokens"
            );
        }

        const auto& info = engine.model_info();
        std::cout << std::fixed << std::setprecision(3)
                  << "{\n"
                  << "  \"engine\": \"jetsonfabric-native\",\n"
                  << "  \"architecture\": \"" << info.architecture << "\",\n"
                  << "  \"source_sha256\": \"" << info.source_sha256 << "\",\n"
                  << "  \"requested_backend\": \""
                  << jetsonfabric::native::backend_name(args.backend) << "\",\n"
                  << "  \"backend\": \"" << info.compute_backend << "\",\n"
                  << "  \"requested_prefill_attention_kernel\": \""
                  << jetsonfabric::native::attention_kernel_name(
                         args.prefill_attention_kernel
                     )
                  << "\",\n"
                  << "  \"prefill_attention_kernel\": \""
                  << jetsonfabric::native::attention_kernel_name(
                         engine.prefill_attention_kernel()
                     )
                  << "\",\n"
                  << "  \"requested_decode_attention_kernel\": \""
                  << jetsonfabric::native::attention_kernel_name(
                         args.decode_attention_kernel
                     )
                  << "\",\n"
                  << "  \"decode_attention_kernel\": \""
                  << jetsonfabric::native::attention_kernel_name(
                         engine.decode_attention_kernel()
                     )
                  << "\",\n"
                  << "  \"device\": \"" << info.compute_device << "\",\n"
                  << "  \"layer_count\": " << info.layer_count << ",\n"
                  << "  \"embedding_length\": " << info.embedding_length << ",\n"
                  << "  \"vocabulary_size\": " << info.vocabulary_size << ",\n"
                  << "  \"weight_bytes\": " << info.weight_bytes << ",\n"
                  << "  \"prompt_tokens\": ";
        print_tokens(args.tokens);
        std::cout << ",\n  \"sampled_tokens\": ";
        print_tokens(sampled_tokens);
        std::cout << ",\n  \"alternate_sampled_tokens\": ";
        print_tokens(alternate_sampled_tokens);
        std::cout << ",\n"
                  << "  \"warmups\": " << args.warmups << ",\n"
                  << "  \"iterations\": " << args.iterations << ",\n"
                  << "  \"session_policy\": \""
                  << session_policy_name(args.session_policy) << "\",\n"
                  << "  \"load_ms\": " << engine.load_time_ms() << ",\n"
                  << "  \"ttft_ms\": ";
        print_samples(ttft);
        std::cout << ",\n  \"itl_ms\": ";
        print_samples(itl);
        std::cout << ",\n  \"decode_tokens_per_second\": ";
        print_samples(decode_throughput);
        std::cout << ",\n  \"end_to_end_tokens_per_second\": ";
        print_samples(end_to_end_throughput);
        std::cout << ",\n  \"alternate_ttft_ms\": ";
        print_samples(alternate_ttft);
        std::cout << ",\n  \"alternate_end_to_end_tokens_per_second\": ";
        print_samples(alternate_end_to_end_throughput);
        std::cout << ",\n  \"prefill_planning_ms\": ";
        print_samples(prefill_graph_build);
        std::cout << ",\n  \"prefill_allocation_ms\": ";
        print_samples(prefill_graph_allocation);
        std::cout << ",\n  \"prefill_host_input_preparation_ms\": ";
        print_samples(prefill_host_input_preparation);
        std::cout << ",\n  \"prefill_compute_ms\": ";
        print_samples(prefill_compute);
        std::cout << ",\n  \"prefill_output_read_ms\": ";
        print_samples(prefill_output_read);
        std::cout << ",\n"
                  << "  \"ttft_p50_ms\": " << median(ttft) << ",\n"
                  << "  \"itl_p50_ms\": " << median(itl) << ",\n"
                  << "  \"decode_tokens_per_second_p50\": "
                  << median(decode_throughput) << ",\n"
                  << "  \"end_to_end_tokens_per_second_p50\": "
                  << median(end_to_end_throughput) << ",\n"
                  << "  \"prefill_plan_reuse_count\": "
                  << prefill_plan_reuse_count << ",\n"
                  << "  \"prefill_attention_backend_verified_count\": "
                  << prefill_attention_backend_verified_count << ",\n"
                  << "  \"prefill_planning_p50_ms\": "
                  << median(prefill_graph_build) << ",\n"
                  << "  \"prefill_allocation_p50_ms\": "
                  << median(prefill_graph_allocation) << ",\n"
                  << "  \"prefill_host_input_preparation_p50_ms\": "
                  << median(prefill_host_input_preparation) << ",\n"
                  << "  \"prefill_compute_p50_ms\": "
                  << median(prefill_compute) << ",\n"
                  << "  \"prefill_output_read_p50_ms\": "
                  << median(prefill_output_read) << ",\n"
                  << "  \"kv_cache_bytes\": " << buffers.kv_cache_bytes << ",\n"
                  << "  \"prefill_scratch_bytes\": "
                  << buffers.prefill_scratch_bytes << ",\n"
                  << "  \"decode_scratch_bytes\": "
                  << buffers.decode_scratch_bytes << ",\n"
                  << "  \"prefill_host_input_bytes\": "
                  << buffers.prefill_host_input_bytes << ",\n"
                  << "  \"kv_cache\": " << (args.incremental ? "true" : "false") << ",\n"
                  << "  \"decode_policy\": \""
                  << (args.incremental ? "incremental" : "full_prefix_recompute")
                  << "\"\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-native-inference-bench: " << error.what() << '\n';
        return 1;
    }
}
