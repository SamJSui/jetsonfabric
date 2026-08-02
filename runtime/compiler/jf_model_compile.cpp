#include "format.h"
#include "jetsonfabric/engine.h"
#include "ggml.h"
#include "gguf.h"
#include "sha256.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <linux/fs.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TensorDescriptor {
    std::string name;
    std::uint32_t type = 0;
    std::uint32_t rank = 0;
    std::array<std::uint64_t, 4> shape{};
    std::int32_t layer = -1;
    std::uint64_t source_offset = 0;
    std::uint64_t size = 0;
    std::uint64_t package_offset = 0;
    std::uint32_t storage_block_elements = 0;
    std::uint32_t storage_block_bytes = 0;
};

struct Segment {
    std::uint32_t kind = JF_SEGMENT_SHARED;
    std::int32_t layer = -1;
    std::string path;
    std::vector<TensorDescriptor> tensors;
    std::uint64_t tensor_bytes = 0;
    std::uint64_t file_size = 0;
    std::array<std::uint8_t, 32> sha256{};
};

struct Arguments {
    fs::path input;
    fs::path output;
};

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, std::string_view label) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw std::runtime_error(std::string(label) + " overflows uint64");
    }
    return left + right;
}

class SourceFile {
public:
    explicit SourceFile(const fs::path& path) {
        descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (descriptor_ < 0) {
            throw std::invalid_argument("could not open input GGUF: " + path.string());
        }
        struct stat metadata{};
        if (fstat(descriptor_, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::invalid_argument("input is not a regular GGUF file: " + path.string());
        }
        if (flock(descriptor_, LOCK_SH) != 0) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("could not lock input GGUF");
        }
        size_ = static_cast<std::uint64_t>(metadata.st_size);
        fd_path_ = "/proc/self/fd/" + std::to_string(descriptor_);
    }

    ~SourceFile() {
        if (descriptor_ >= 0) {
            (void) flock(descriptor_, LOCK_UN);
            close(descriptor_);
        }
    }

    SourceFile(const SourceFile&) = delete;
    SourceFile& operator=(const SourceFile&) = delete;

    const fs::path& path() const { return fd_path_; }
    std::uint64_t size() const { return size_; }

private:
    int descriptor_ = -1;
    std::uint64_t size_ = 0;
    fs::path fd_path_;
};

class StagingDirectory {
public:
    explicit StagingDirectory(const fs::path& output) : output_(output) {
        fs::path parent = output.parent_path();
        if (parent.empty()) {
            parent = ".";
        }
        if (!fs::is_directory(parent)) {
            throw std::invalid_argument("output parent is not a directory: " + parent.string());
        }
        staging_ = parent / (output.filename().string() + ".tmp." + std::to_string(getpid()));
        if (!fs::create_directory(staging_)) {
            throw std::runtime_error("could not create exclusive staging directory: " + staging_.string());
        }
    }

    ~StagingDirectory() {
        if (!published_) {
            std::error_code ignored;
            fs::remove_all(staging_, ignored);
        }
    }

    const fs::path& path() const { return staging_; }

    void publish() {
        const long result = syscall(
            SYS_renameat2,
            AT_FDCWD,
            staging_.c_str(),
            AT_FDCWD,
            output_.c_str(),
            RENAME_NOREPLACE
        );
        if (result != 0) {
            const int saved_errno = errno;
            throw std::runtime_error(
                "filesystem does not support atomic no-replace package publication: " +
                output_.string() + ": " + strerror(saved_errno)
            );
        }
        published_ = true;
        fsync_directory(output_.parent_path().empty() ? fs::path(".") : output_.parent_path());
    }

private:
    static void fsync_directory(const fs::path& path) {
        const int descriptor = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (descriptor < 0 || fsync(descriptor) != 0) {
            const int saved_errno = errno;
            if (descriptor >= 0) close(descriptor);
            throw std::runtime_error("could not sync output directory: " + std::string(strerror(saved_errno)));
        }
        close(descriptor);
    }

    fs::path output_;
    fs::path staging_;
    bool published_ = false;
};

Segment make_segment(std::uint32_t kind, std::int32_t layer, std::string path) {
    Segment segment;
    segment.kind = kind;
    segment.layer = layer;
    segment.path = std::move(path);
    return segment;
}

class GgufContextDeleter {
public:
    void operator()(gguf_context * context) const {
        gguf_free(context);
    }
};

using GgufContext = std::unique_ptr<gguf_context, GgufContextDeleter>;

class OpenSslApi {
public:
    using ContextNew = void * (*)();
    using ContextFree = void (*)(void *);
    using Sha256 = const void * (*)();
    using DigestInit = int (*)(void *, const void *, void *);
    using DigestUpdate = int (*)(void *, const void *, std::size_t);
    using DigestFinal = int (*)(void *, unsigned char *, unsigned int *);

    OpenSslApi() {
        library_ = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            library_ = dlopen("libcrypto.so", RTLD_NOW | RTLD_LOCAL);
        }
        available_ = library_ != nullptr &&
            load(context_new_, "EVP_MD_CTX_new") &&
            load(context_free_, "EVP_MD_CTX_free") &&
            load(sha256_, "EVP_sha256") &&
            load(digest_init_, "EVP_DigestInit_ex") &&
            load(digest_update_, "EVP_DigestUpdate") &&
            load(digest_final_, "EVP_DigestFinal_ex");
    }

    OpenSslApi(const OpenSslApi&) = delete;
    OpenSslApi& operator=(const OpenSslApi&) = delete;

    bool available() const { return available_; }
    ContextNew context_new() const { return context_new_; }
    ContextFree context_free() const { return context_free_; }
    Sha256 sha256() const { return sha256_; }
    DigestInit digest_init() const { return digest_init_; }
    DigestUpdate digest_update() const { return digest_update_; }
    DigestFinal digest_final() const { return digest_final_; }

private:
    template <typename Function>
    bool load(Function& function, const char * name) {
        void * symbol = dlsym(library_, name);
        static_assert(sizeof(function) == sizeof(symbol));
        std::memcpy(&function, &symbol, sizeof(function));
        return function != nullptr;
    }

    void * library_ = nullptr;
    bool available_ = false;
    ContextNew context_new_ = nullptr;
    ContextFree context_free_ = nullptr;
    Sha256 sha256_ = nullptr;
    DigestInit digest_init_ = nullptr;
    DigestUpdate digest_update_ = nullptr;
    DigestFinal digest_final_ = nullptr;
};

OpenSslApi& openssl_api() {
    static OpenSslApi api;
    return api;
}

class Sha256Digest {
public:
    Sha256Digest() {
        OpenSslApi& api = openssl_api();
        if (api.available()) {
            openssl_context_ = api.context_new()();
            if (openssl_context_ == nullptr ||
                api.digest_init()(openssl_context_, api.sha256()(), nullptr) != 1) {
                if (openssl_context_ != nullptr) {
                    api.context_free()(openssl_context_);
                }
                throw std::runtime_error("OpenSSL could not initialize SHA-256");
            }
        } else {
            jf_sha256_init(&fallback_context_);
        }
    }

    ~Sha256Digest() {
        if (openssl_context_ != nullptr) {
            openssl_api().context_free()(openssl_context_);
        }
    }

    Sha256Digest(const Sha256Digest&) = delete;
    Sha256Digest& operator=(const Sha256Digest&) = delete;

    void update(const void * data, std::size_t size) {
        if (size == 0) {
            return;
        }
        if (openssl_context_ != nullptr) {
            if (openssl_api().digest_update()(openssl_context_, data, size) != 1) {
                throw std::runtime_error("OpenSSL SHA-256 update failed");
            }
        } else {
            jf_sha256_update(&fallback_context_, data, size);
        }
    }

    std::array<std::uint8_t, 32> finish() {
        std::array<std::uint8_t, 32> digest{};
        if (openssl_context_ != nullptr) {
            unsigned int size = 0;
            if (openssl_api().digest_final()(openssl_context_, digest.data(), &size) != 1 ||
                size != digest.size()) {
                throw std::runtime_error("OpenSSL SHA-256 finalization failed");
            }
            openssl_api().context_free()(openssl_context_);
            openssl_context_ = nullptr;
        } else {
            jf_sha256_final(&fallback_context_, digest.data());
        }
        return digest;
    }

private:
    void * openssl_context_ = nullptr;
    jf_sha256_context fallback_context_{};
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw std::runtime_error("model package size overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
}

void append_i32(std::vector<std::uint8_t>& output, std::int32_t value) {
    append_u32(output, static_cast<std::uint32_t>(value));
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
}

void pad_to(std::vector<std::uint8_t>& output, std::size_t alignment) {
    while (output.size() % alignment != 0) {
        output.push_back(0);
    }
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create " + path.string());
    }
    output.write(
        reinterpret_cast<const char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );
    if (!output) {
        throw std::runtime_error("could not write " + path.string());
    }
    output.close();
    if (!output) {
        throw std::runtime_error("could not finish " + path.string());
    }
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        const int saved_errno = errno;
        if (descriptor >= 0) close(descriptor);
        throw std::runtime_error("could not sync " + path.string() + ": " + strerror(saved_errno));
    }
    close(descriptor);
}

std::array<std::uint8_t, 32> hash_stream(std::istream& input) {
    Sha256Digest digest;
    std::vector<char> buffer(8U << 20U);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            digest.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while hashing input stream");
    }
    return digest.finish();
}

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(digest.size() * 2);
    for (const std::uint8_t byte : digest) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

Arguments parse_arguments(int argc, char ** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            std::cout << "Usage: jf-model-compile --input model.gguf --output model.jfm\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--input") {
            arguments.input = value;
        } else if (option == "--output") {
            arguments.output = value;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.input.empty() || arguments.output.empty()) {
        throw std::invalid_argument("--input and --output are required");
    }
    return arguments;
}

std::uint64_t unsigned_metadata(const gguf_context * context, std::string_view key) {
    const std::string key_string(key);
    const std::int64_t id = gguf_find_key(context, key_string.c_str());
    if (id < 0) {
        throw std::runtime_error("GGUF metadata is missing " + key_string);
    }
    switch (gguf_get_kv_type(context, id)) {
    case GGUF_TYPE_UINT8:
        return gguf_get_val_u8(context, id);
    case GGUF_TYPE_UINT16:
        return gguf_get_val_u16(context, id);
    case GGUF_TYPE_UINT32:
        return gguf_get_val_u32(context, id);
    case GGUF_TYPE_UINT64:
        return gguf_get_val_u64(context, id);
    case GGUF_TYPE_INT8: {
        const std::int8_t value = gguf_get_val_i8(context, id);
        if (value >= 0) return static_cast<std::uint64_t>(value);
        break;
    }
    case GGUF_TYPE_INT16: {
        const std::int16_t value = gguf_get_val_i16(context, id);
        if (value >= 0) return static_cast<std::uint64_t>(value);
        break;
    }
    case GGUF_TYPE_INT32: {
        const std::int32_t value = gguf_get_val_i32(context, id);
        if (value >= 0) return static_cast<std::uint64_t>(value);
        break;
    }
    case GGUF_TYPE_INT64: {
        const std::int64_t value = gguf_get_val_i64(context, id);
        if (value >= 0) return static_cast<std::uint64_t>(value);
        break;
    }
    default:
        break;
    }
    throw std::runtime_error("GGUF metadata " + key_string + " is not a non-negative integer");
}

std::string string_metadata(const gguf_context * context, std::string_view key) {
    const std::string key_string(key);
    const std::int64_t id = gguf_find_key(context, key_string.c_str());
    if (id < 0 || gguf_get_kv_type(context, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF metadata is missing string " + key_string);
    }
    return gguf_get_val_str(context, id);
}

std::optional<std::uint32_t> tensor_layer(std::string_view name) {
    constexpr std::string_view prefix = "blk.";
    if (!name.starts_with(prefix)) {
        return std::nullopt;
    }
    std::size_t cursor = prefix.size();
    if (cursor >= name.size() || name[cursor] < '0' || name[cursor] > '9') {
        throw std::runtime_error("invalid block tensor name: " + std::string(name));
    }
    std::uint64_t value = 0;
    while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
        value = value * 10U + static_cast<unsigned>(name[cursor] - '0');
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("block index overflows in tensor " + std::string(name));
        }
        ++cursor;
    }
    if (cursor >= name.size() || name[cursor] != '.') {
        throw std::runtime_error("invalid block tensor name: " + std::string(name));
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t tensor_rank(const std::int64_t * dimensions) {
    std::uint32_t rank = 1;
    for (std::uint32_t index = 1; index < 4; ++index) {
        if (dimensions[index] > 1) {
            rank = index + 1;
        }
    }
    return rank;
}

std::uint32_t stable_tensor_type(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32: return JF_TENSOR_F32;
    case GGML_TYPE_F16: return JF_TENSOR_F16;
    case GGML_TYPE_Q4_0: return JF_TENSOR_Q4_0;
    case GGML_TYPE_Q4_1: return JF_TENSOR_Q4_1;
    case GGML_TYPE_Q5_0: return JF_TENSOR_Q5_0;
    case GGML_TYPE_Q5_1: return JF_TENSOR_Q5_1;
    case GGML_TYPE_Q8_0: return JF_TENSOR_Q8_0;
    case GGML_TYPE_Q8_1: return JF_TENSOR_Q8_1;
    case GGML_TYPE_Q2_K: return JF_TENSOR_Q2_K;
    case GGML_TYPE_Q3_K: return JF_TENSOR_Q3_K;
    case GGML_TYPE_Q4_K: return JF_TENSOR_Q4_K;
    case GGML_TYPE_Q5_K: return JF_TENSOR_Q5_K;
    case GGML_TYPE_Q6_K: return JF_TENSOR_Q6_K;
    case GGML_TYPE_Q8_K: return JF_TENSOR_Q8_K;
    case GGML_TYPE_IQ2_XXS: return JF_TENSOR_IQ2_XXS;
    case GGML_TYPE_IQ2_XS: return JF_TENSOR_IQ2_XS;
    case GGML_TYPE_IQ3_XXS: return JF_TENSOR_IQ3_XXS;
    case GGML_TYPE_IQ1_S: return JF_TENSOR_IQ1_S;
    case GGML_TYPE_IQ4_NL: return JF_TENSOR_IQ4_NL;
    case GGML_TYPE_IQ3_S: return JF_TENSOR_IQ3_S;
    case GGML_TYPE_IQ2_S: return JF_TENSOR_IQ2_S;
    case GGML_TYPE_IQ4_XS: return JF_TENSOR_IQ4_XS;
    case GGML_TYPE_I8: return JF_TENSOR_I8;
    case GGML_TYPE_I16: return JF_TENSOR_I16;
    case GGML_TYPE_I32: return JF_TENSOR_I32;
    case GGML_TYPE_I64: return JF_TENSOR_I64;
    case GGML_TYPE_F64: return JF_TENSOR_F64;
    case GGML_TYPE_IQ1_M: return JF_TENSOR_IQ1_M;
    case GGML_TYPE_BF16: return JF_TENSOR_BF16;
    case GGML_TYPE_TQ1_0: return JF_TENSOR_TQ1_0;
    case GGML_TYPE_TQ2_0: return JF_TENSOR_TQ2_0;
    case GGML_TYPE_MXFP4: return JF_TENSOR_MXFP4;
    case GGML_TYPE_NVFP4: return JF_TENSOR_NVFP4;
    case GGML_TYPE_Q1_0: return JF_TENSOR_Q1_0;
    case GGML_TYPE_Q2_0: return JF_TENSOR_Q2_0;
    case GGML_TYPE_COUNT: break;
    }
    throw std::runtime_error("unsupported GGML tensor type " + std::to_string(type));
}

std::string layer_path(std::uint32_t layer) {
    char path[32];
    std::snprintf(path, sizeof(path), "layer-%05u.jfs", layer);
    return path;
}

std::vector<Segment> collect_segments(
    const gguf_context * context,
    std::uint32_t layer_count,
    std::uint64_t source_data_offset,
    std::uint64_t source_size
) {
    Segment shared = make_segment(JF_SEGMENT_SHARED, -1, "shared.jfs");
    Segment input = make_segment(JF_SEGMENT_INPUT, -1, "input.jfs");
    Segment output = make_segment(JF_SEGMENT_OUTPUT, -1, "output.jfs");
    std::vector<Segment> layers;
    bool has_output_projection = false;
    layers.reserve(layer_count);
    for (std::uint32_t layer = 0; layer < layer_count; ++layer) {
        layers.push_back(make_segment(
            JF_SEGMENT_LAYER,
            static_cast<std::int32_t>(layer),
            layer_path(layer)
        ));
    }

    const std::int64_t tensor_count = gguf_get_n_tensors(context);
    for (std::int64_t tensor_id = 0; tensor_id < tensor_count; ++tensor_id) {
        TensorDescriptor tensor;
        tensor.name = gguf_get_tensor_name(context, tensor_id);
        if (tensor.name.empty() || tensor.name.size() > 1024) {
            throw std::runtime_error("GGUF tensor name length is unsupported");
        }
        const ggml_type source_type = gguf_get_tensor_type(context, tensor_id);
        tensor.type = stable_tensor_type(source_type);
        const std::int64_t block_elements = ggml_blck_size(source_type);
        const std::size_t block_bytes = ggml_type_size(source_type);
        std::uint32_t canonical_block_elements = 0;
        std::uint32_t canonical_block_bytes = 0;
        const jf_status layout_status = jf_tensor_type_layout(
            tensor.type,
            &canonical_block_elements,
            &canonical_block_bytes
        );
        if (layout_status.code != JF_STATUS_OK ||
            block_elements != canonical_block_elements || block_bytes != canonical_block_bytes) {
            throw std::runtime_error(
                "pinned GGML tensor layout disagrees with JFM v2 for " + tensor.name
            );
        }
        tensor.storage_block_elements = canonical_block_elements;
        tensor.storage_block_bytes = canonical_block_bytes;
        const std::int64_t * dimensions = gguf_get_tensor_ne(context, tensor_id);
        tensor.rank = tensor_rank(dimensions);
        for (std::uint32_t dimension = 0; dimension < 4; ++dimension) {
            if (dimensions[dimension] <= 0) {
                throw std::runtime_error("tensor has an invalid shape: " + tensor.name);
            }
            tensor.shape[dimension] = static_cast<std::uint64_t>(dimensions[dimension]);
        }
        tensor.size = gguf_get_tensor_size(context, tensor_id);
        tensor.source_offset = checked_add(
            source_data_offset,
            gguf_get_tensor_offset(context, tensor_id),
            "GGUF tensor offset"
        );
        if (tensor.source_offset > source_size || tensor.size > source_size - tensor.source_offset) {
            throw std::runtime_error("tensor exceeds the source GGUF: " + tensor.name);
        }

        if (const std::optional<std::uint32_t> layer = tensor_layer(tensor.name)) {
            if (*layer >= layer_count) {
                throw std::runtime_error("tensor block exceeds model block count: " + tensor.name);
            }
            tensor.layer = static_cast<std::int32_t>(*layer);
            layers[*layer].tensors.push_back(std::move(tensor));
        } else if (tensor.name.starts_with("token_embd.")) {
            input.tensors.push_back(std::move(tensor));
        } else if (tensor.name.starts_with("output.") || tensor.name.starts_with("output_norm.")) {
            has_output_projection = has_output_projection || tensor.name == "output.weight";
            output.tensors.push_back(std::move(tensor));
        } else {
            shared.tensors.push_back(std::move(tensor));
        }
    }

    std::vector<Segment> segments;
    if (!shared.tensors.empty()) segments.push_back(std::move(shared));
    if (input.tensors.empty()) {
        throw std::runtime_error("GGUF contains no input embedding tensors");
    }
    segments.push_back(std::move(input));
    for (Segment& layer : layers) {
        if (layer.tensors.empty()) {
            throw std::runtime_error("GGUF contains no tensors for " + layer.path);
        }
        segments.push_back(std::move(layer));
    }
    if (!has_output_projection) {
        throw std::runtime_error(
            "tied output embeddings are not supported by JFM v2; output.weight is required"
        );
    }
    segments.push_back(std::move(output));
    return segments;
}

void copy_range(
    std::ifstream& source,
    std::ofstream& destination,
    Sha256Digest& destination_hash,
    std::uint64_t offset,
    std::uint64_t size
) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(offset));
    if (!source) {
        throw std::runtime_error("could not seek to GGUF tensor data");
    }
    std::vector<char> buffer(8U << 20U);
    while (size > 0) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(size, buffer.size())
        );
        source.read(buffer.data(), static_cast<std::streamsize>(count));
        if (source.gcount() != static_cast<std::streamsize>(count)) {
            throw std::runtime_error("GGUF tensor data is truncated");
        }
        destination.write(buffer.data(), static_cast<std::streamsize>(count));
        if (!destination) {
            throw std::runtime_error("could not write model segment");
        }
        destination_hash.update(buffer.data(), count);
        size -= count;
    }
}

void write_and_hash(
    std::ofstream& destination,
    Sha256Digest& hash,
    const void * data,
    std::size_t size
) {
    if (size == 0) {
        return;
    }
    destination.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!destination) {
        throw std::runtime_error("could not write model segment");
    }
    hash.update(data, size);
}

void write_segment(const fs::path& source_path, const fs::path& output_path, Segment& segment) {
    std::uint64_t index_size = 0;
    for (const TensorDescriptor& tensor : segment.tensors) {
        const std::uint64_t record_size = checked_add(
            JF_TENSOR_RECORD_SIZE,
            tensor.name.size(),
            "tensor index record"
        );
        index_size = checked_add(index_size, align_up(record_size, 8), "segment index");
    }
    std::uint64_t data_cursor = align_up(
        checked_add(JF_SEGMENT_HEADER_SIZE, index_size, "segment header"),
        JF_FORMAT_ALIGNMENT
    );
    segment.tensor_bytes = 0;
    for (TensorDescriptor& tensor : segment.tensors) {
        tensor.package_offset = data_cursor;
        data_cursor = align_up(
            checked_add(data_cursor, tensor.size, "segment tensor payload"),
            JF_FORMAT_ALIGNMENT
        );
        segment.tensor_bytes = checked_add(
            segment.tensor_bytes,
            tensor.size,
            "segment tensor byte total"
        );
    }
    segment.file_size = data_cursor;

    std::vector<std::uint8_t> index;
    index.insert(index.end(), std::begin(JF_SEGMENT_MAGIC), std::end(JF_SEGMENT_MAGIC));
    append_u32(index, JF_MODEL_FORMAT_VERSION);
    append_u32(index, static_cast<std::uint32_t>(segment.tensors.size()));
    append_u64(index, index_size);
    append_u64(index, segment.tensor_bytes);
    index.resize(JF_SEGMENT_HEADER_SIZE, 0);
    for (const TensorDescriptor& tensor : segment.tensors) {
        append_u32(index, tensor.type);
        append_u32(index, tensor.rank);
        append_i32(index, tensor.layer);
        append_u32(index, 0);
        for (const std::uint64_t dimension : tensor.shape) {
            append_u64(index, dimension);
        }
        append_u64(index, tensor.package_offset);
        append_u64(index, tensor.size);
        append_u32(index, static_cast<std::uint32_t>(tensor.name.size()));
        append_u32(index, tensor.storage_block_elements);
        append_u32(index, tensor.storage_block_bytes);
        append_u32(index, 0);
        index.insert(index.end(), tensor.name.begin(), tensor.name.end());
        pad_to(index, 8);
    }
    if (index.size() != JF_SEGMENT_HEADER_SIZE + index_size) {
        throw std::runtime_error("internal segment index size mismatch");
    }
    index.resize(static_cast<std::size_t>(align_up(index.size(), JF_FORMAT_ALIGNMENT)), 0);

    std::ifstream source(source_path, std::ios::binary);
    std::ofstream destination(output_path, std::ios::binary | std::ios::trunc);
    if (!source || !destination) {
        throw std::runtime_error("could not open source or destination for " + segment.path);
    }
    Sha256Digest segment_hash;
    write_and_hash(destination, segment_hash, index.data(), index.size());
    for (const TensorDescriptor& tensor : segment.tensors) {
        const std::uint64_t position = static_cast<std::uint64_t>(destination.tellp());
        if (position > tensor.package_offset) {
            throw std::runtime_error("internal tensor offset overlap");
        }
        const std::vector<char> padding(static_cast<std::size_t>(tensor.package_offset - position), 0);
        write_and_hash(destination, segment_hash, padding.data(), padding.size());
        copy_range(source, destination, segment_hash, tensor.source_offset, tensor.size);
    }
    const std::uint64_t position = static_cast<std::uint64_t>(destination.tellp());
    const std::vector<char> padding(static_cast<std::size_t>(segment.file_size - position), 0);
    write_and_hash(destination, segment_hash, padding.data(), padding.size());
    destination.close();
    if (!destination) {
        throw std::runtime_error("could not finish model segment " + segment.path);
    }
    segment.sha256 = segment_hash.finish();
    const int descriptor = open(output_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        const int saved_errno = errno;
        if (descriptor >= 0) close(descriptor);
        throw std::runtime_error(
            "could not sync model segment " + segment.path + ": " + strerror(saved_errno)
        );
    }
    close(descriptor);
}

void write_metadata_segment(
    const fs::path& source_path,
    const fs::path& output_path,
    std::uint64_t metadata_size,
    Segment& segment
) {
    if (metadata_size == 0) {
        throw std::runtime_error("GGUF metadata prefix is empty");
    }
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream destination(output_path, std::ios::binary | std::ios::trunc);
    if (!source || !destination) {
        throw std::runtime_error("could not open GGUF metadata source or destination");
    }
    Sha256Digest digest;
    copy_range(source, destination, digest, 0, metadata_size);
    destination.close();
    if (!destination) {
        throw std::runtime_error("could not finish GGUF metadata segment");
    }
    segment.file_size = metadata_size;
    segment.sha256 = digest.finish();
    const int descriptor = open(output_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        const int saved_errno = errno;
        if (descriptor >= 0) close(descriptor);
        throw std::runtime_error("could not sync GGUF metadata segment: " + std::string(strerror(saved_errno)));
    }
    close(descriptor);
}

void write_manifest(
    const fs::path& path,
    std::uint32_t layer_count,
    std::uint64_t source_size,
    const std::array<std::uint8_t, 32>& source_sha256,
    const std::vector<Segment>& segments
) {
    std::vector<std::uint8_t> manifest;
    manifest.insert(manifest.end(), std::begin(JF_MANIFEST_MAGIC), std::end(JF_MANIFEST_MAGIC));
    append_u32(manifest, JF_MODEL_FORMAT_VERSION);
    append_u32(manifest, layer_count);
    append_u64(manifest, source_size);
    manifest.insert(manifest.end(), source_sha256.begin(), source_sha256.end());
    append_u32(manifest, static_cast<std::uint32_t>(segments.size()));
    append_u32(manifest, 0);
    for (const Segment& segment : segments) {
        append_u32(manifest, segment.kind);
        append_i32(manifest, segment.layer);
        append_u64(manifest, segment.tensors.size());
        append_u64(manifest, segment.tensor_bytes);
        append_u64(manifest, segment.file_size);
        manifest.insert(manifest.end(), segment.sha256.begin(), segment.sha256.end());
        append_u32(manifest, static_cast<std::uint32_t>(segment.path.size()));
        append_u32(manifest, 0);
        manifest.insert(manifest.end(), segment.path.begin(), segment.path.end());
    }
    write_bytes(path, manifest);
}

void sync_directory(const fs::path& path) {
    const int descriptor = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        const int saved_errno = errno;
        if (descriptor >= 0) close(descriptor);
        throw std::runtime_error("could not sync package directory: " + std::string(strerror(saved_errno)));
    }
    close(descriptor);
}

void compile_model(const Arguments& arguments) {
    if (fs::exists(arguments.output)) {
        throw std::invalid_argument("output already exists: " + arguments.output.string());
    }
    SourceFile source(arguments.input);
    const std::uint64_t source_size = source.size();
    if (source_size > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("input GGUF is too large for this compiler build");
    }

    gguf_init_params params{.no_alloc = true, .ctx = nullptr};
    GgufContext context(gguf_init_from_file(source.path().c_str(), params));
    if (!context) {
        throw std::runtime_error("could not parse GGUF metadata");
    }
    const std::int64_t split_count_key = gguf_find_key(context.get(), "split.count");
    if (split_count_key >= 0 && unsigned_metadata(context.get(), "split.count") > 1) {
        throw std::runtime_error("split GGUF inputs are not supported; merge the shards first");
    }

    const std::string architecture = string_metadata(context.get(), "general.architecture");
    const std::uint64_t layer_count_value = unsigned_metadata(
        context.get(),
        architecture + ".block_count"
    );
    if (layer_count_value == 0 || layer_count_value > 4096U) {
        throw std::runtime_error("GGUF block count is invalid");
    }
    const std::uint32_t layer_count = static_cast<std::uint32_t>(layer_count_value);

    std::ifstream hash_input(source.path(), std::ios::binary);
    if (!hash_input) {
        throw std::runtime_error("could not reopen locked GGUF for hashing");
    }
    const std::array<std::uint8_t, 32> source_sha256 = hash_stream(hash_input);
    const std::uint64_t metadata_size = gguf_get_data_offset(context.get());
    if (metadata_size == 0 || metadata_size > source_size) {
        throw std::runtime_error("GGUF metadata offset is outside the source file");
    }
    std::vector<Segment> segments = collect_segments(
        context.get(),
        layer_count,
        metadata_size,
        source_size
    );
    segments.insert(
        segments.begin(),
        make_segment(JF_SEGMENT_METADATA, -1, "metadata.gguf")
    );

    StagingDirectory package(arguments.output);
    write_metadata_segment(
        source.path(),
        package.path() / segments.front().path,
        metadata_size,
        segments.front()
    );
    for (std::size_t index = 1; index < segments.size(); ++index) {
        write_segment(
            source.path(),
            package.path() / segments[index].path,
            segments[index]
        );
    }

    std::ifstream final_hash_input(source.path(), std::ios::binary);
    if (!final_hash_input || hash_stream(final_hash_input) != source_sha256) {
        throw std::runtime_error("input GGUF changed while the package was being compiled");
    }
    write_manifest(
        package.path() / "manifest.jfm",
        layer_count,
        source_size,
        source_sha256,
        segments
    );
    sync_directory(package.path());

    std::uint64_t tensor_count = 0;
    std::uint64_t tensor_bytes = 0;
    std::uint64_t package_bytes = fs::file_size(package.path() / "manifest.jfm");
    for (const Segment& segment : segments) {
        tensor_count = checked_add(tensor_count, segment.tensors.size(), "package tensor count");
        tensor_bytes = checked_add(tensor_bytes, segment.tensor_bytes, "package tensor bytes");
        package_bytes = checked_add(package_bytes, segment.file_size, "package file bytes");
    }
    package.publish();
    std::cout
        << "{\n"
        << "  \"format\": \"jfm-v2\",\n"
        << "  \"architecture\": \"" << architecture << "\",\n"
        << "  \"sha256_backend\": \""
        << (openssl_api().available() ? "openssl" : "portable") << "\",\n"
        << "  \"source_sha256\": \"" << hex_digest(source_sha256) << "\",\n"
        << "  \"layer_count\": " << layer_count << ",\n"
        << "  \"tensor_count\": " << tensor_count << ",\n"
        << "  \"tensor_bytes\": " << tensor_bytes << ",\n"
        << "  \"package_bytes\": " << package_bytes << "\n"
        << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        compile_model(parse_arguments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jf-model-compile: " << error.what() << '\n';
        return 1;
    }
}
