#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string model_path;
    std::string prompt;
    std::uint32_t max_tokens = 1;
    std::uint32_t warmups = 0;
    std::uint32_t iterations = 1;
    int n_gpu_layers = 0;
    int threads = 1;
};

std::uint32_t parse_u32(const std::string& value, const char * name) {
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed > UINT32_MAX) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::uint32_t>(parsed);
}

Arguments parse_args(int argc, char ** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--model") args.model_path = value;
        else if (option == "--prompt") args.prompt = value;
        else if (option == "--max-tokens") args.max_tokens = parse_u32(value, option.c_str());
        else if (option == "--warmups") args.warmups = parse_u32(value, option.c_str());
        else if (option == "--iterations") args.iterations = parse_u32(value, option.c_str());
        else if (option == "--n-gpu-layers") {
            args.n_gpu_layers = static_cast<int>(parse_u32(value, option.c_str()));
        } else if (option == "--threads") {
            args.threads = static_cast<int>(parse_u32(value, option.c_str()));
        } else throw std::invalid_argument("unknown option " + option);
    }
    if (args.model_path.empty() || args.prompt.empty() || args.max_tokens == 0 ||
        args.iterations == 0 || args.threads <= 0) {
        throw std::invalid_argument(
            "--model, --prompt, positive --max-tokens, --iterations, and --threads are required"
        );
    }
    return args;
}

std::vector<llama_token> tokenize(const llama_vocab * vocabulary, const std::string& prompt) {
    int count = llama_tokenize(
        vocabulary, prompt.data(), static_cast<std::int32_t>(prompt.size()),
        nullptr, 0, true, true
    );
    if (count >= 0) throw std::runtime_error("llama.cpp did not return required token capacity");
    std::vector<llama_token> tokens(static_cast<std::size_t>(-count));
    count = llama_tokenize(
        vocabulary, prompt.data(), static_cast<std::int32_t>(prompt.size()),
        tokens.data(), static_cast<std::int32_t>(tokens.size()), true, true
    );
    if (count <= 0) throw std::runtime_error("llama.cpp could not tokenize prompt");
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

void print_tokens(const std::vector<llama_token>& tokens) {
    std::cout << '[';
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << ']';
}

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}

void print_samples(const std::vector<double>& samples) {
    std::cout << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << std::fixed << std::setprecision(3) << samples[index];
    }
    std::cout << ']';
}

struct GenerationTiming {
    std::vector<llama_token> sampled_tokens;
    double ttft_ms = 0.0;
    double itl_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    double end_to_end_tokens_per_second = 0.0;
};

GenerationTiming generate(
    llama_context * context,
    llama_sampler * sampler,
    const std::vector<llama_token>& prompt,
    std::uint32_t max_tokens
) {
    llama_memory_clear(llama_get_memory(context), false);
    llama_sampler_reset(sampler);
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()),
        static_cast<std::int32_t>(prompt.size())
    );
    GenerationTiming timing;
    double total_ms = 0.0;
    double decode_ms = 0.0;
    for (std::uint32_t index = 0; index < max_tokens; ++index) {
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(context, batch) != 0) {
            throw std::runtime_error("llama.cpp oracle decode failed");
        }
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        const double token_ms = elapsed_ms(start, std::chrono::steady_clock::now());
        if (index == 0) timing.ttft_ms = token_ms;
        else decode_ms += token_ms;
        total_ms += token_ms;
        timing.sampled_tokens.push_back(token);
        batch = llama_batch_get_one(&timing.sampled_tokens.back(), 1);
    }
    if (max_tokens > 1) {
        timing.itl_ms = decode_ms / static_cast<double>(max_tokens - 1);
        timing.decode_tokens_per_second = 1000.0 / timing.itl_ms;
    }
    timing.end_to_end_tokens_per_second =
        1000.0 * static_cast<double>(max_tokens) / total_ms;
    return timing;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = args.n_gpu_layers;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(args.model_path.c_str(), model_params),
            llama_model_free
        );
        if (!model) throw std::runtime_error("llama.cpp could not load oracle model");
        const llama_vocab * vocabulary = llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt_tokens = tokenize(vocabulary, args.prompt);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(
            prompt_tokens.size() + args.max_tokens + 8U
        );
        context_params.n_batch = static_cast<std::uint32_t>(prompt_tokens.size());
        context_params.n_ubatch = std::min(context_params.n_batch, 512U);
        context_params.n_threads = args.threads;
        context_params.n_threads_batch = args.threads;
        context_params.no_perf = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params),
            llama_free
        );
        if (!context) throw std::runtime_error("llama.cpp could not create oracle context");
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(),
            llama_sampler_free
        );

        for (std::uint32_t index = 0; index < args.warmups; ++index) {
            (void) generate(context.get(), sampler.get(), prompt_tokens, args.max_tokens);
        }

        std::vector<double> ttft;
        std::vector<double> itl;
        std::vector<double> decode_throughput;
        std::vector<double> end_to_end_throughput;
        std::vector<llama_token> sampled;
        for (std::uint32_t index = 0; index < args.iterations; ++index) {
            const GenerationTiming timing = generate(
                context.get(), sampler.get(), prompt_tokens, args.max_tokens
            );
            if (index == 0) sampled = timing.sampled_tokens;
            if (timing.sampled_tokens != sampled) {
                throw std::runtime_error("llama.cpp greedy output changed between iterations");
            }
            ttft.push_back(timing.ttft_ms);
            itl.push_back(timing.itl_ms);
            decode_throughput.push_back(timing.decode_tokens_per_second);
            end_to_end_throughput.push_back(timing.end_to_end_tokens_per_second);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "{\n  \"engine\": \"llama.cpp-oracle\",\n  \"prompt_tokens\": ";
        print_tokens(prompt_tokens);
        std::cout << ",\n  \"sampled_tokens\": ";
        print_tokens(sampled);
        std::cout << ",\n"
                  << "  \"warmups\": " << args.warmups << ",\n"
                  << "  \"iterations\": " << args.iterations << ",\n"
                  << "  \"ttft_ms\": ";
        print_samples(ttft);
        std::cout << ",\n  \"itl_ms\": ";
        print_samples(itl);
        std::cout << ",\n  \"decode_tokens_per_second\": ";
        print_samples(decode_throughput);
        std::cout << ",\n  \"end_to_end_tokens_per_second\": ";
        print_samples(end_to_end_throughput);
        std::cout << ",\n"
                  << "  \"ttft_p50_ms\": " << median(ttft) << ",\n"
                  << "  \"itl_p50_ms\": " << median(itl) << ",\n"
                  << "  \"decode_tokens_per_second_p50\": "
                  << median(decode_throughput) << ",\n"
                  << "  \"end_to_end_tokens_per_second_p50\": "
                  << median(end_to_end_throughput) << ",\n"
                  << "  \"kv_cache\": true,\n"
                  << "  \"decode_policy\": \"incremental\"\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-llama-greedy-oracle: " << error.what() << '\n';
        return 1;
    }
}
