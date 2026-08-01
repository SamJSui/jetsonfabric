#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        ggml_free(context);
    }
};

void fill(ggml_tensor * tensor, float value) {
    float * data = static_cast<float *>(tensor->data);
    for (std::int64_t index = 0; index < ggml_nelements(tensor); ++index) {
        data[index] = value;
    }
}

ggml_tensor * add_vector(
    ggml_context * context,
    gguf_context * gguf,
    const char * name,
    std::int64_t elements,
    float value
) {
    ggml_tensor * tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, elements);
    ggml_set_name(tensor, name);
    fill(tensor, value);
    gguf_add_tensor(gguf, tensor);
    return tensor;
}

ggml_tensor * add_matrix(
    ggml_context * context,
    gguf_context * gguf,
    const char * name,
    std::int64_t columns,
    std::int64_t rows,
    float value
) {
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, columns, rows);
    ggml_set_name(tensor, name);
    fill(tensor, value);
    gguf_add_tensor(gguf, tensor);
    return tensor;
}

void write_fixture(const std::string& path) {
    std::vector<std::uint8_t> memory(1U << 20U);
    ggml_init_params params{
        .mem_size = memory.size(),
        .mem_buffer = memory.data(),
        .no_alloc = false,
    };
    ContextDeleter context_deleter;
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        throw std::runtime_error("could not create GGML fixture context");
    }

    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        context_deleter(context);
        throw std::runtime_error("could not create GGUF fixture context");
    }
    gguf_set_val_str(gguf, "general.architecture", "qwen2");
    gguf_set_val_u32(gguf, "qwen2.block_count", 2);
    (void) add_matrix(context, gguf, "token_embd.weight", 8, 4, 1.0F);
    (void) add_vector(context, gguf, "blk.0.attn_norm.weight", 4, 2.0F);
    (void) add_matrix(context, gguf, "blk.0.attn_q.weight", 4, 4, 3.0F);
    (void) add_vector(context, gguf, "blk.1.attn_norm.weight", 4, 4.0F);
    (void) add_matrix(context, gguf, "blk.1.attn_q.weight", 4, 4, 5.0F);
    (void) add_vector(context, gguf, "output_norm.weight", 4, 6.0F);
    (void) add_matrix(context, gguf, "output.weight", 4, 8, 7.0F);

    if (!gguf_write_to_file(gguf, path.c_str(), false)) {
        gguf_free(gguf);
        context_deleter(context);
        throw std::runtime_error("could not write GGUF fixture");
    }
    gguf_free(gguf);
    context_deleter(context);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("output path is required");
        }
        write_fixture(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-model-fixture: " << error.what() << '\n';
        return 1;
    }
}
