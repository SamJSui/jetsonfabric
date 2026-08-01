#pragma once

#include "jetsonfabric/engine.h"
#include "jetsonfabric/native_inference.hpp"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <cstddef>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace jetsonfabric::native {

class TensorStore {
public:
    TensorStore(
        const std::string& package_path,
        Backend backend,
        int threads
    );

    ggml_backend_t backend() const { return backend_.get(); }
    ggml_tensor * find(const std::string& name) const;
    ggml_tensor * require(const std::string& name) const;
    std::uint32_t vocabulary_size() const;
    std::uint64_t weight_bytes() const { return weight_bytes_; }
    std::span<const std::byte> gguf_metadata() const { return metadata_; }
    const std::string& source_sha256() const { return source_sha256_; }
    const std::string& backend_name() const { return backend_name_; }
    const std::string& device_name() const { return device_name_; }

private:
    static ggml_backend_ptr create_backend(Backend backend, int threads);
    void create_tensors(jf_model * model);
    void copy_tensors(jf_model * model);

    ggml_backend_ptr backend_;
    ggml_context_ptr context_;
    ggml_backend_buffer_ptr buffer_;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    std::vector<std::byte> metadata_;
    std::string source_sha256_;
    std::string backend_name_;
    std::string device_name_;
    std::uint64_t weight_bytes_ = 0;
};

} // namespace jetsonfabric::native
