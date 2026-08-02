#include "jetsonfabric/native_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Arguments {
    std::string package_path;
    jetsonfabric::native::Backend backend = jetsonfabric::native::Backend::Cuda;
    std::vector<std::int32_t> tokens;
    std::uint32_t decode_steps = 4;
    int threads = 1;
};

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
        else if (option == "--decode-steps") {
            const unsigned long parsed = std::stoul(value);
            if (parsed == 0 || parsed > UINT32_MAX) {
                throw std::invalid_argument("--decode-steps must be positive");
            }
            args.decode_steps = static_cast<std::uint32_t>(parsed);
        }
        else if (option == "--threads") args.threads = std::stoi(value);
        else if (option == "--backend" && value == "cpu") {
            args.backend = jetsonfabric::native::Backend::Cpu;
        } else if (option == "--backend" && value == "cuda") {
            args.backend = jetsonfabric::native::Backend::Cuda;
        } else {
            throw std::invalid_argument("unknown option or backend: " + option + " " + value);
        }
    }
    if (args.package_path.empty() || args.tokens.empty() || args.threads <= 0) {
        throw std::invalid_argument(
            "--package, --tokens, and a positive --threads value are required"
        );
    }
    return args;
}

struct TraceResult {
    std::vector<std::vector<float>> logits;
    std::vector<std::int32_t> forced_tokens;
};

TraceResult reference_trace(const Arguments& args) {
    jetsonfabric::native::NativeEngine engine(
        args.package_path,
        jetsonfabric::native::EngineOptions{
            .backend = args.backend,
            .prefill_attention_kernel = jetsonfabric::native::AttentionKernel::Unfused,
            .decode_attention_kernel = jetsonfabric::native::AttentionKernel::Unfused,
            .threads = args.threads,
        }
    );
    const jetsonfabric::native::GenerationResult generation = engine.generate(
        args.tokens, args.decode_steps + 1U
    );
    std::vector<std::int32_t> forced_tokens(
        generation.sampled_tokens.begin(),
        generation.sampled_tokens.begin() + args.decode_steps
    );
    return {
        .logits = engine.forced_decode_logits(args.tokens, forced_tokens),
        .forced_tokens = std::move(forced_tokens),
    };
}

std::vector<std::vector<float>> candidate_trace(
    const Arguments& args,
    std::span<const std::int32_t> forced_tokens
) {
    jetsonfabric::native::NativeEngine engine(
        args.package_path,
        jetsonfabric::native::EngineOptions{
            .backend = args.backend,
            .prefill_attention_kernel = jetsonfabric::native::AttentionKernel::Flash,
            .decode_attention_kernel = jetsonfabric::native::AttentionKernel::Flash,
            .threads = args.threads,
        }
    );
    return engine.forced_decode_logits(args.tokens, forced_tokens);
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end()))
    );
}

struct Comparison {
    std::size_t reference_argmax = 0;
    std::size_t candidate_argmax = 0;
    double normalized_rmse = 0.0;
    double cosine_similarity = 0.0;
    double max_absolute_error = 0.0;
};

Comparison compare_logits(
    const std::vector<float>& reference,
    const std::vector<float>& candidate
) {
    if (reference.empty() || candidate.size() != reference.size()) {
        throw std::runtime_error("attention parity logits have inconsistent shapes");
    }
    double squared_error = 0.0;
    double reference_squared = 0.0;
    double candidate_squared = 0.0;
    double dot_product = 0.0;
    double max_absolute_error = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(reference[index]) || !std::isfinite(candidate[index])) {
            throw std::runtime_error("attention parity logits contain a non-finite value");
        }
        const double difference = static_cast<double>(candidate[index]) - reference[index];
        squared_error += difference * difference;
        reference_squared += static_cast<double>(reference[index]) * reference[index];
        candidate_squared += static_cast<double>(candidate[index]) * candidate[index];
        dot_product += static_cast<double>(reference[index]) * candidate[index];
        max_absolute_error = std::max(max_absolute_error, std::abs(difference));
    }
    return {
        .reference_argmax = argmax(reference),
        .candidate_argmax = argmax(candidate),
        .normalized_rmse = std::sqrt(
            squared_error / std::max(reference_squared, 1.0e-30)
        ),
        .cosine_similarity = dot_product / std::max(
            std::sqrt(reference_squared * candidate_squared), 1.0e-30
        ),
        .max_absolute_error = max_absolute_error,
    };
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        const TraceResult reference = reference_trace(args);
        const std::vector<std::vector<float>> candidate = candidate_trace(
            args, reference.forced_tokens
        );
        if (candidate.size() != reference.logits.size()) {
            throw std::runtime_error("attention parity traces have inconsistent lengths");
        }
        std::vector<Comparison> comparisons;
        comparisons.reserve(reference.logits.size());
        bool all_argmax_match = true;
        double max_normalized_rmse = 0.0;
        double min_cosine_similarity = 1.0;
        double max_absolute_error = 0.0;
        for (std::size_t index = 0; index < reference.logits.size(); ++index) {
            comparisons.push_back(compare_logits(reference.logits[index], candidate[index]));
            const Comparison& comparison = comparisons.back();
            all_argmax_match = all_argmax_match &&
                comparison.reference_argmax == comparison.candidate_argmax;
            max_normalized_rmse = std::max(
                max_normalized_rmse, comparison.normalized_rmse
            );
            min_cosine_similarity = std::min(
                min_cosine_similarity, comparison.cosine_similarity
            );
            max_absolute_error = std::max(
                max_absolute_error, comparison.max_absolute_error
            );
        }

        std::cout << std::fixed << std::setprecision(9)
                  << "{\n"
                  << "  \"engine\": \"jetsonfabric-native\",\n"
                  << "  \"backend\": \""
                  << jetsonfabric::native::backend_name(args.backend) << "\",\n"
                  << "  \"reference_prefill_attention_kernel\": \"unfused\",\n"
                  << "  \"candidate_prefill_attention_kernel\": \"flash\",\n"
                  << "  \"reference_decode_attention_kernel\": \"unfused\",\n"
                  << "  \"candidate_decode_attention_kernel\": \"flash\",\n"
                  << "  \"prompt_token_count\": " << args.tokens.size() << ",\n"
                  << "  \"decode_steps\": " << args.decode_steps << ",\n"
                  << "  \"vocabulary_size\": " << reference.logits.front().size() << ",\n"
                  << "  \"argmax_match\": "
                  << (all_argmax_match ? "true" : "false") << ",\n"
                  << "  \"normalized_rmse\": " << max_normalized_rmse << ",\n"
                  << "  \"cosine_similarity\": " << min_cosine_similarity << ",\n"
                  << "  \"max_absolute_error\": " << max_absolute_error << ",\n"
                  << "  \"steps\": [\n";
        for (std::size_t index = 0; index < comparisons.size(); ++index) {
            const Comparison& comparison = comparisons[index];
            std::cout << "    {\"phase\": \""
                      << (index == 0 ? "prefill" : "decode") << "\", "
                      << "\"decode_step\": " << (index == 0 ? 0 : index) << ", "
                      << "\"reference_argmax\": " << comparison.reference_argmax << ", "
                      << "\"candidate_argmax\": " << comparison.candidate_argmax << ", "
                      << "\"normalized_rmse\": " << comparison.normalized_rmse << ", "
                      << "\"cosine_similarity\": " << comparison.cosine_similarity << ", "
                      << "\"max_absolute_error\": " << comparison.max_absolute_error << "}"
                      << (index + 1 == comparisons.size() ? "\n" : ",\n");
        }
        std::cout << "  ]\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-native-attention-parity: " << error.what() << '\n';
        return 1;
    }
}
