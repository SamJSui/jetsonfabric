#include "adapters/llama_cpp_adapter.hpp"

#include "llama.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jetsonfabric::runtime::adapters {
namespace {

void free_context(llama_context* context) {
    llama_free(context);
}

void free_sampler(llama_sampler* sampler) {
    llama_sampler_free(sampler);
}

using ContextPtr = std::unique_ptr<llama_context, decltype(&free_context)>;
using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&free_sampler)>;

} // namespace

struct LlamaCppAdapter::Impl {
    Impl(std::shared_ptr<LlamaCppModel> model_in, int ctx_size_in, int threads_in)
        : model(std::move(model_in)), ctx_size(ctx_size_in), threads(threads_in) {
        if (!model) {
            throw std::invalid_argument("llama.cpp model is required");
        }
    }

    GenerateResponse generate(const GenerateRequest& request) {
        std::lock_guard<std::mutex> lock(mutex);
        const int max_tokens = std::max(1, request.max_tokens);
        std::vector<std::int32_t> prompt_tokens = model->tokenize(request.prompt);
        ContextPtr context = create_context(prompt_tokens, max_tokens);
        SamplerPtr sampler = create_sampler();

        GenerateResponse response;
        response.prompt_tokens = static_cast<int>(prompt_tokens.size());
        decode_tokens(context.get(), sampler.get(), prompt_tokens, max_tokens, response);
        return response;
    }

    ContextPtr create_context(const std::vector<std::int32_t>& prompt_tokens, int max_tokens) const {
        const int required = static_cast<int>(prompt_tokens.size()) + max_tokens + 8;
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(std::max(ctx_size, required));
        params.n_batch = static_cast<std::uint32_t>(std::max<int>(1, prompt_tokens.size()));
        params.no_perf = true;
        if (threads > 0) {
            params.n_threads = threads;
            params.n_threads_batch = threads;
        }
        llama_context* raw = llama_init_from_model(model->raw_model(), params);
        if (raw == nullptr) {
            throw std::runtime_error("failed to create llama.cpp context");
        }
        return ContextPtr(raw, free_context);
    }

    static SamplerPtr create_sampler() {
        llama_sampler_chain_params params = llama_sampler_chain_default_params();
        params.no_perf = true;
        llama_sampler* raw = llama_sampler_chain_init(params);
        if (raw == nullptr) {
            throw std::runtime_error("failed to create llama.cpp sampler");
        }
        llama_sampler_chain_add(raw, llama_sampler_init_greedy());
        return SamplerPtr(raw, free_sampler);
    }

    void decode_tokens(
        llama_context* context,
        llama_sampler* sampler,
        std::vector<std::int32_t>& prompt_tokens,
        int max_tokens,
        GenerateResponse& response
    ) const {
        std::vector<llama_token> llama_tokens(prompt_tokens.begin(), prompt_tokens.end());
        llama_batch batch = llama_batch_get_one(llama_tokens.data(), static_cast<int32_t>(llama_tokens.size()));
        llama_token next_token = LLAMA_TOKEN_NULL;
        for (int index = 0; index < max_tokens; ++index) {
            const int status = llama_decode(context, batch);
            if (status != 0) {
                throw std::runtime_error("llama_decode failed with code " + std::to_string(status));
            }
            next_token = llama_sampler_sample(sampler, context, -1);
            if (model->is_end_token(next_token)) {
                break;
            }
            response.token_ids.push_back(next_token);
            response.text += model->token_piece(next_token);
            response.completion_tokens += 1;
            batch = llama_batch_get_one(&next_token, 1);
        }
    }

    std::shared_ptr<LlamaCppModel> model;
    int ctx_size = 4096;
    int threads = 0;
    std::mutex mutex;
};

LlamaCppAdapter::LlamaCppAdapter(LlamaCppConfig config)
    : LlamaCppAdapter(
          std::make_shared<LlamaCppModel>(LlamaCppModelConfig{
              .model_path = std::move(config.model_path),
              .n_gpu_layers = config.n_gpu_layers,
          }),
          config.ctx_size,
          config.threads) {}

LlamaCppAdapter::LlamaCppAdapter(std::shared_ptr<LlamaCppModel> model, int ctx_size, int threads)
    : impl_(std::make_unique<Impl>(std::move(model), ctx_size, threads)) {}

LlamaCppAdapter::~LlamaCppAdapter() = default;

GenerateResponse LlamaCppAdapter::generate(const GenerateRequest& request) {
    return impl_->generate(request);
}

std::shared_ptr<LlamaCppModel> LlamaCppAdapter::model() const {
    return impl_->model;
}

} // namespace jetsonfabric::runtime::adapters
