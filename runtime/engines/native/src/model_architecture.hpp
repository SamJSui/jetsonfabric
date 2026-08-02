#pragma once

#include "jetsonfabric/native_inference.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace jetsonfabric::native {

class TensorStore;

struct PrefillResult {
    std::int32_t token = -1;
    PrefillMetrics metrics;
};

class InferenceSession {
public:
    virtual ~InferenceSession() = default;

    virtual std::size_t capacity() const = 0;
    virtual void reset() = 0;
    virtual std::vector<float> prefill_logits(
        std::span<const std::int32_t> tokens
    ) = 0;
    virtual PrefillResult prefill_greedy(
        std::span<const std::int32_t> tokens
    ) = 0;
    virtual std::int32_t decode_greedy(std::int32_t token) = 0;
    virtual ExecutionBufferMetrics execution_buffers() const = 0;
};

class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;

    virtual const ModelInfo& model_info() const = 0;
    virtual std::unique_ptr<InferenceSession> create_session(
        TensorStore& tensors,
        std::size_t capacity
    ) const = 0;
};

std::unique_ptr<ModelArchitecture> create_architecture(const TensorStore& tensors);

} // namespace jetsonfabric::native
