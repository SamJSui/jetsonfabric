#include "adapters/llama_cpp_adapter.hpp"

#include "llama.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
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
    int n = llama_token_to_piece(vocab, token, small.data(), static_cast<int32_t>(small.size()), 0, true);

    if (n >= 0) {
        return std::string(small.data(), static_cast<std::size_t>(n));
    }

    std::string large(static_cast<std::size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, token, large.data(), static_cast<int32_t>(large.size()), 0, true);

    if (n < 0) {
        throw std::runtime_error("failed to convert llama token to text");
    }

    large.resize(static_cast<std::size_t>(n));
    return large;
}

void free_context(llama_context* ctx) {
    llama_free(ctx);
}

void free_sampler(llama_sampler* sampler) {
    llama_sampler_free(sampler);
}

using ContextPtr = std::unique_ptr<llama_context, decltype(&free_context)>;
using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&free_sampler)>;

} // namespace

struct LlamaCppAdapter::Impl {
    explicit Impl(LlamaCppConfig config_in)
        : config(std::move(config_in)) {
        load_model();
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

        const std::string prompt = normalize_prompt(request.prompt);
        const int max_tokens = std::max(1, request.max_tokens);
        std::vector<llama_token> prompt_tokens = tokenize_prompt(prompt);

        ContextPtr ctx = create_context(prompt_tokens, max_tokens);
        SamplerPtr sampler = create_sampler();

        int completion_tokens = 0;
        std::string output = decode_tokens(ctx.get(), sampler.get(), prompt_tokens, max_tokens, completion_tokens);

        return GenerateResponse{
            .text = output,
            .prompt_tokens = static_cast<int>(prompt_tokens.size()),
            .completion_tokens = completion_tokens,
        };
    }

    void load_model() {
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

    static std::string normalize_prompt(const std::string& prompt) {
        if (prompt.empty()) {
            return "Hello";
        }
        return prompt;
    }

    int count_prompt_tokens(const std::string& prompt) const {
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
        return n_prompt;
    }

    std::vector<llama_token> tokenize_prompt(const std::string& prompt) const {
        std::vector<llama_token> tokens(static_cast<std::size_t>(count_prompt_tokens(prompt)));
        const int actual = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            true
        );
        if (actual < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }
        tokens.resize(static_cast<std::size_t>(actual));
        return tokens;
    }

    ContextPtr create_context(const std::vector<llama_token>& prompt_tokens, int max_tokens) const {
        const int required_ctx = static_cast<int>(prompt_tokens.size()) + max_tokens + 8;
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(std::max(config.ctx_size, required_ctx));
        params.n_batch = static_cast<uint32_t>(std::max<int>(1, prompt_tokens.size()));
        params.no_perf = true;
        if (config.threads > 0) {
            params.n_threads = config.threads;
            params.n_threads_batch = config.threads;
        }
        llama_context* ctx = llama_init_from_model(model, params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create llama.cpp context");
        }
        return ContextPtr(ctx, free_context);
    }

    SamplerPtr create_sampler() const {
        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        params.no_perf = true;
        llama_sampler* sampler = llama_sampler_chain_init(params);
        if (sampler == nullptr) {
            throw std::runtime_error("failed to create llama.cpp sampler");
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        return SamplerPtr(sampler, free_sampler);
    }

    std::string decode_tokens(
        llama_context* ctx,
        llama_sampler* sampler,
        std::vector<llama_token>& prompt_tokens,
        int max_tokens,
        int& completion_tokens
    ) const {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
        llama_token next_token = LLAMA_TOKEN_NULL;
        std::string output;

        for (int i = 0; i < max_tokens; ++i) {
            decode_batch(ctx, batch);
            next_token = llama_sampler_sample(sampler, ctx, -1);
            if (is_end_token(next_token)) {
                break;
            }
            output += token_to_piece(vocab, next_token);
            completion_tokens += 1;
            batch = llama_batch_get_one(&next_token, 1);
        }
        return output;
    }

    static void decode_batch(llama_context* ctx, llama_batch batch) {
        const int decode_result = llama_decode(ctx, batch);
        if (decode_result != 0) {
            throw std::runtime_error(
                "llama_decode failed with code " + std::to_string(decode_result)
            );
        }
    }

    bool is_end_token(llama_token token) const {
        return token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token);
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
