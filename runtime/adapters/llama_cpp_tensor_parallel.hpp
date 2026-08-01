#pragma once

#include "tensor_parallel/device_mesh.hpp"

#include <cstddef>
#include <memory>

struct llama_model_params;

namespace jetsonfabric::runtime::adapters {

// Maps JetsonFabric's tensor device mesh to llama.cpp's quantized meta-device
// tensor sharding. RPC registration is contained here so other engines can use
// a different collective backend without changing runtime orchestration.
class LlamaCppTensorParallel final {
public:
    explicit LlamaCppTensorParallel(tensor_parallel::DeviceMesh mesh);
    ~LlamaCppTensorParallel();

    LlamaCppTensorParallel(const LlamaCppTensorParallel&) = delete;
    LlamaCppTensorParallel& operator=(const LlamaCppTensorParallel&) = delete;

    void configure(llama_model_params& params) const;
    std::size_t world_size() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jetsonfabric::runtime::adapters
