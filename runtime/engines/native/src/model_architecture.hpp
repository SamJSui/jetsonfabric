#pragma once

#include "jetsonfabric/native_inference.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace jetsonfabric::native {

class TensorStore;

class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;

    virtual const ModelInfo& model_info() const = 0;
    virtual std::vector<float> logits(
        TensorStore& tensors,
        std::span<const std::int32_t> tokens
    ) const = 0;
};

std::unique_ptr<ModelArchitecture> create_architecture(const TensorStore& tensors);

} // namespace jetsonfabric::native
