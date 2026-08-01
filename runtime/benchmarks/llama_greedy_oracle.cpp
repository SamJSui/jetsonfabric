#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
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
        else if (option == "--n-gpu-layers") {
            args.n_gpu_layers = static_cast<int>(parse_u32(value, option.c_str()));
        } else if (option == "--threads") {
            args.threads = static_cast<int>(parse_u32(value, option.c_str()));
        } else throw std::invalid_argument("unknown option " + option);
    }
    if (args.model_path.empty() || args.prompt.empty() || args.max_tokens == 0 || args.threads <= 0) {
        throw std::invalid_argument(
            "--model, --prompt, positive --max-tokens, and --threads are required"
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

        std::vector<llama_token> sampled;
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(), static_cast<std::int32_t>(prompt_tokens.size())
        );
        for (std::uint32_t index = 0; index < args.max_tokens; ++index) {
            if (llama_decode(context.get(), batch) != 0) {
                throw std::runtime_error("llama.cpp oracle decode failed");
            }
            const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
            sampled.push_back(token);
            batch = llama_batch_get_one(&sampled.back(), 1);
        }
        llama_synchronize(context.get());

        std::cout << "{\n  \"engine\": \"llama.cpp-oracle\",\n  \"prompt_tokens\": ";
        print_tokens(prompt_tokens);
        std::cout << ",\n  \"sampled_tokens\": ";
        print_tokens(sampled);
        std::cout << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-llama-greedy-oracle: " << error.what() << '\n';
        return 1;
    }
}
