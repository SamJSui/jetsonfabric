#pragma once

#include "adapters/llama_cpp_model.hpp"
#include "inference/stage.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace jetsonfabric::runtime::adapters {

struct LlamaCppStageConfig {
    std::shared_ptr<LlamaCppModel> model;
    int ctx_size = 4096;
    int threads = 0;
    inference::StagePosition position;
    inference::LayerRange layers;
};

// Owns persistent llama.cpp contexts keyed by JetsonFabric session ID. Each
// context executes only the configured transformer-layer range and retains the
// corresponding llama.cpp memory state across decode steps.
class LlamaCppStageAdapter final {
public:
    explicit LlamaCppStageAdapter(LlamaCppStageConfig config);
    ~LlamaCppStageAdapter();

    LlamaCppStageAdapter(const LlamaCppStageAdapter&) = delete;
    LlamaCppStageAdapter& operator=(const LlamaCppStageAdapter&) = delete;

    inference::ExecutionResult execute(const inference::StageInput& input) const;
    void close_session(const std::string& session_id) const;
    std::size_t session_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
