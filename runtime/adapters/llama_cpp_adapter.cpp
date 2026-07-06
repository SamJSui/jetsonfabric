#include "adapters/llama_cpp_adapter.hpp"

#include "llama.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

std::once_flag backend_init_once;

void ensure_backend_loaded() {
    std::call_once(backend_init_once, [] {
        ggml_backend_load_all();
    });
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    std::array<char, 256> small{};

    int n = llama_token_to_piece(
        vocab,
        token,
        small.data(),
        static_cast<int32_t>(small.size()),
        0,
        true
    );

    if (n >= 0) {
        return std::string(small.data(), static_cast<std::size_t>(n));
    }

    std::string large(static_cast<std::size_t>(-n), '\0');

    n = llama_token_to_piece(
        vocab,
        token,
        large.data(),
        static_cast<int32_t>(large.size()),
        0,
        true
    );

    if (n < 0) {
        throw std::runtime_error("failed to convert llama token to text");
    }

    large.resize(static_cast<std::size_t>(n));
    return large;
}

} // namespace

struct LlamaCppAdapter::Impl {
    explicit Impl(LlamaCppConfig config_in)
        : config(std::move(config_in)) {
        if (config.model_path.empty()) {
            throw std::runtime_error("llama.cpp model path is required");
        }

        ensure_backend_loaded();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers;

        model = llama_model_load_from_file(config.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load llama.cpp model: " + config.model_path);
        }

        vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            throw std::runtime_error("failed to get llama.cpp vocab");
        }
    }

    ~Impl() {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    GenerateResponse generate(const GenerateRequest& request) {
        std::lock_guard<std::mutex> lock(mu);

        const std::string prompt = request.prompt.empty() ? "Hello" : request.prompt;
        const int max_tokens = std::max(1, request.max_tokens);

        int n_prompt = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            nullptr,
            0,
            true,
            true
        );

        if (n_prompt < 0) {
            n_prompt = -n_prompt;
        }

        if (n_prompt <= 0) {
            throw std::runtime_error("failed to count prompt tokens");
        }

        std::vector<llama_token> prompt_tokens(static_cast<std::size_t>(n_prompt));

        const int actual_prompt_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true
        );

        if (actual_prompt_tokens < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }

        prompt_tokens.resize(static_cast<std::size_t>(actual_prompt_tokens));

        const int required_ctx =
            static_cast<int>(prompt_tokens.size()) + max_tokens + 8;

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = static_cast<uint32_t>(
            std::max(config.ctx_size, required_ctx)
        );
        ctx_params.n_batch = static_cast<uint32_t>(
            std::max<int>(1, prompt_tokens.size())
        );
        ctx_params.no_perf = true;

        if (config.threads > 0) {
            ctx_params.n_threads = config.threads;
            ctx_params.n_threads_batch = config.threads;
        }

        llama_context* ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create llama.cpp context");
        }

        auto free_ctx = [&] {
            if (ctx != nullptr) {
                llama_free(ctx);
                ctx = nullptr;
            }
        };

        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = true;

        llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
        if (sampler == nullptr) {
            free_ctx();
            throw std::runtime_error("failed to create llama.cpp sampler");
        }

        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        auto free_sampler = [&] {
            if (sampler != nullptr) {
                llama_sampler_free(sampler);
                sampler = nullptr;
            }
        };

        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size())
        );

        std::string output;
        int completion_tokens = 0;

        for (int i = 0; i < max_tokens; ++i) {
            const int decode_result = llama_decode(ctx, batch);
            if (decode_result != 0) {
                free_sampler();
                free_ctx();
                throw std::runtime_error(
                    "llama_decode failed with code " + std::to_string(decode_result)
                );
            }

            llama_token token = llama_sampler_sample(sampler, ctx, -1);

            if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
                break;
            }

            output += token_to_piece(vocab, token);
            completion_tokens += 1;

            batch = llama_batch_get_one(&token, 1);
        }

        free_sampler();
        free_ctx();

        return GenerateResponse{
            .text = output,
            .prompt_tokens = static_cast<int>(prompt_tokens.size()),
            .completion_tokens = completion_tokens,
        };
    }

    LlamaCppConfig config;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    std::mutex mu;
};

LlamaCppAdapter::LlamaCppAdapter(LlamaCppConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LlamaCppAdapter::~LlamaCppAdapter() = default;

GenerateResponse LlamaCppAdapter::generate(const GenerateRequest& request) {
    return impl_->generate(request);
}

} // namespace jetsonfabric::runtime::adapters