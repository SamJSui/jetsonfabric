#pragma once

#include "activation/activation_codec.hpp"
#include "deployment/model_manager.hpp"
#include "engine/engine.hpp"
#include "engine/inference_engine_factory.hpp"
#include "transport/stage_transport.hpp"
#include "worker/config.hpp"

#include <memory>

namespace jetsonfabric::runtime {

// RuntimeService exposes the runtime HTTP contract while ModelManager owns
// resident deployment epochs and their execution components.
class RuntimeService final : public RuntimeAPI {
public:
    explicit RuntimeService(Config config);
    RuntimeService(
        Config config,
        std::shared_ptr<const InferenceEngineFactory> engine_factory,
        std::shared_ptr<const transport::StageTransport> stage_transport
    );
    RuntimeService(
        Config config,
        std::shared_ptr<const InferenceEngineFactory> engine_factory,
        std::shared_ptr<const transport::StageTransport> stage_transport,
        std::shared_ptr<const activation::ActivationCodec> activation_codec
    );

    void shutdown() override;
    std::string runtime_name() const override;
    std::string engine_name() const override;
    ExecutionMode execution_mode() const override;
    std::string model() const override;
    std::string stage_transport_name() const override;
    std::string activation_encoding() const override;
    std::string kv_cache_type() const override;
    int ubatch_size() const override;
    int parallel_sessions() const override;
    int decode_batch_size() const override;
    std::string speculative_draft() const override;
    int speculative_max_tokens() const override;

    RuntimeResponse deployment_status() const override;
    RuntimeResponse load_deployment(const std::string& request_body) override;
    RuntimeResponse activate_deployment(const std::string& request_body) override;
    RuntimeResponse drain_deployment(const std::string& request_body) override;
    RuntimeResponse unload_deployment(const std::string& request_body) override;
    RuntimeResponse chat_completion(const std::string& request_body) const override;
    RuntimeResponse generate(
        const std::string& request_body,
        const GenerationEventSink& sink
    ) const override;
    RuntimeResponse run_stage(const std::string& request_body) const override;

private:
    Config config_;
    std::shared_ptr<const InferenceEngineFactory> engine_factory_;
    std::shared_ptr<const transport::StageTransport> stage_transport_;
    std::shared_ptr<const activation::ActivationCodec> activation_codec_;
    deployment::ModelManager model_manager_;
};

} // namespace jetsonfabric::runtime
