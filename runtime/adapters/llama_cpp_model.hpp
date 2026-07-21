#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace jetsonfabric::runtime::adapters {

struct LlamaCppModelConfig {
    std::string model_path;
    int n_gpu_layers = 0;
    int layer_start = 0;
    int layer_end = 0;
};

class LlamaCppModel final {
public:
    explicit LlamaCppModel(LlamaCppModelConfig config);
    ~LlamaCppModel();

    LlamaCppModel(const LlamaCppModel&) = delete;
    LlamaCppModel& operator=(const LlamaCppModel&) = delete;

    llama_model* raw_model() const;
    const llama_vocab* raw_vocab() const;

    int n_embd() const;
    int n_layer() const;
    int n_vocab() const;
    const std::string& architecture() const;
    int loaded_layer_start() const;
    int loaded_layer_end() const;
    std::uint64_t resident_weight_bytes() const;
    std::uint64_t total_weight_bytes() const;
    std::uint64_t resident_tensor_count() const;

    std::vector<std::int32_t> tokenize(std::string_view text, bool add_special = true) const;
    std::string token_piece(std::int32_t token) const;
    bool is_end_token(std::int32_t token) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
