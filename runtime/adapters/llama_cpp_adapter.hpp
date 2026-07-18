#pragma once

#include "adapters/llama_cpp_model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jetsonfabric::runtime::adapters {

struct LlamaCppConfig {
    std::string model_path;
    int ctx_size = 4096;
    int n_gpu_layers = 999;
    int threads = 0;
};

struct GenerateRequest {
    std::string prompt;
    int max_tokens = 128;
};

struct GenerateResponse {
    std::string text;
    std::vector<std::int32_t> token_ids;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

class LlamaCppAdapter final {
public:
    explicit LlamaCppAdapter(LlamaCppConfig config);
    LlamaCppAdapter(std::shared_ptr<LlamaCppModel> model, int ctx_size, int threads);
    ~LlamaCppAdapter();

    LlamaCppAdapter(const LlamaCppAdapter&) = delete;
    LlamaCppAdapter& operator=(const LlamaCppAdapter&) = delete;

    GenerateResponse generate(const GenerateRequest& request);
    std::shared_ptr<LlamaCppModel> model() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
