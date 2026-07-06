#pragma once

#include <memory>
#include <string>

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
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

class LlamaCppAdapter final {
public:
    explicit LlamaCppAdapter(LlamaCppConfig config);
    ~LlamaCppAdapter();

    LlamaCppAdapter(const LlamaCppAdapter&) = delete;
    LlamaCppAdapter& operator=(const LlamaCppAdapter&) = delete;

    GenerateResponse generate(const GenerateRequest& request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters