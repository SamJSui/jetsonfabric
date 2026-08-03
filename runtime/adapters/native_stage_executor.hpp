#pragma once

#include "inference/executor.hpp"
#include "jetsonfabric/native_inference.hpp"
#include "tokenization/tokenizer.hpp"

#include <memory>
#include <string>

namespace jetsonfabric::runtime::adapters {

struct NativeStageConfig {
    std::shared_ptr<native::NativeEngine> engine;
    std::shared_ptr<const tokenization::Tokenizer> tokenizer;
    int ctx_size = 0;
    inference::StagePosition position;
    inference::LayerRange layers;
};

// Adapts the stepwise JetsonFabric stage contract to the native engine. M2 is
// intentionally full-model and one-stage; distributed partitions remain M3.
class NativeStageExecutor final : public inference::Executor {
public:
    explicit NativeStageExecutor(NativeStageConfig config);
    ~NativeStageExecutor() override;

    inference::ExecutionResult execute(const inference::StageInput& input) const override;
    void close_session(const std::string& session_id) const override;
    void rollback_session(const std::string& session_id, int token_count) const override;
    std::size_t session_count() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
