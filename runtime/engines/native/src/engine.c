#include "jetsonfabric/engine.h"

#include "format.h"
#include "sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define JF_MAX_MANIFEST_BYTES (16U * 1024U * 1024U)
#define JF_MAX_LAYERS 4096U
#define JF_MAX_SEGMENTS (JF_MAX_LAYERS + 4U)
#define JF_MAX_TENSORS 1000000U

struct jf_mapping {
    void * address;
    size_t size;
};

struct jf_tensor {
    const char * name;
    size_t name_length;
    uint32_t type;
    uint32_t rank;
    uint32_t storage_block_elements;
    uint32_t storage_block_bytes;
    uint64_t shape[4];
    int32_t layer;
    const void * data;
    uint64_t size;
};

struct jf_model {
    struct jf_mapping * mappings;
    size_t mapping_count;
    struct jf_tensor * tensors;
    size_t tensor_count;
    const void * gguf_metadata;
    size_t gguf_metadata_size;
    uint8_t source_sha256[32];
    jf_model_stats stats;
};

struct manifest_segment {
    uint32_t kind;
    int32_t layer;
    uint64_t tensor_count;
    uint64_t tensor_bytes;
    uint64_t file_size;
    uint8_t sha256[32];
    const char * path;
    uint32_t path_length;
};

static jf_status status_ok(void) {
    const jf_status status = {JF_STATUS_OK, ""};
    return status;
}

static jf_status status_error(jf_status_code code, const char * format, ...) {
    jf_status status = {code, ""};
    va_list args;
    va_start(args, format);
    (void) vsnprintf(status.message, sizeof(status.message), format, args);
    va_end(args);
    return status;
}

static int range_fits(size_t offset, size_t length, size_t size) {
    return offset <= size && length <= size - offset;
}

static int bytes_are_zero(const uint8_t * bytes, size_t size) {
    for (size_t index = 0; index < size; ++index) {
        if (bytes[index] != 0) return 0;
    }
    return 1;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t * result) {
    if (left > UINT64_MAX - right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int multiply_u64(uint64_t left, uint64_t right, uint64_t * result) {
    if (left != 0 && right > UINT64_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int safe_segment_path(const char * path, size_t length) {
    if (length == 0 || length > 255 || path[0] == '.') {
        return 0;
    }
    for (size_t index = 0; index < length; ++index) {
        const char value = path[index];
        if (value == '/' || value == '\\' || value == '\0') {
            return 0;
        }
    }
    return 1;
}

static int same_path(
    const struct manifest_segment * left,
    const struct manifest_segment * right
) {
    return left->path_length == right->path_length &&
        memcmp(left->path, right->path, left->path_length) == 0;
}

static int compare_segment_paths(const void * left_pointer, const void * right_pointer) {
    const struct manifest_segment * left = left_pointer;
    const struct manifest_segment * right = right_pointer;
    const size_t common = left->path_length < right->path_length
        ? left->path_length
        : right->path_length;
    const int result = memcmp(left->path, right->path, common);
    if (result != 0) return result;
    return (left->path_length > right->path_length) - (left->path_length < right->path_length);
}

static jf_status read_manifest(int directory, uint8_t ** data, size_t * size) {
    int descriptor = openat(directory, "manifest.jfm", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        return status_error(JF_STATUS_IO_ERROR, "open manifest.jfm: %s", strerror(errno));
    }

    struct stat metadata;
    if (fstat(descriptor, &metadata) != 0) {
        const int saved_errno = errno;
        close(descriptor);
        return status_error(JF_STATUS_IO_ERROR, "stat manifest.jfm: %s", strerror(saved_errno));
    }
    if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        (uint64_t) metadata.st_size > JF_MAX_MANIFEST_BYTES) {
        close(descriptor);
        return status_error(JF_STATUS_FORMAT_ERROR, "manifest.jfm is not a bounded regular file");
    }

    *size = (size_t) metadata.st_size;
    *data = malloc(*size == 0 ? 1 : *size);
    if (*data == NULL) {
        close(descriptor);
        return status_error(JF_STATUS_OUT_OF_MEMORY, "allocate %zu manifest bytes", *size);
    }

    size_t offset = 0;
    while (offset < *size) {
        const ssize_t count = read(descriptor, *data + offset, *size - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            const int saved_errno = count < 0 ? errno : EIO;
            free(*data);
            *data = NULL;
            close(descriptor);
            return status_error(JF_STATUS_IO_ERROR, "read manifest.jfm: %s", strerror(saved_errno));
        }
        offset += (size_t) count;
    }
    close(descriptor);
    return status_ok();
}

static int segment_selected(const struct manifest_segment * segment, jf_stage_plan plan) {
    switch (segment->kind) {
    case JF_SEGMENT_METADATA:
    case JF_SEGMENT_SHARED:
        return 1;
    case JF_SEGMENT_INPUT:
        return plan.layer_start == 0;
    case JF_SEGMENT_LAYER:
        return segment->layer >= 0 &&
            (uint32_t) segment->layer >= plan.layer_start &&
            (uint32_t) segment->layer < plan.layer_end;
    case JF_SEGMENT_OUTPUT:
        return 0;
    }
    return 0;
}

static void close_model_contents(jf_model * model) {
    if (model == NULL) {
        return;
    }
    for (size_t index = 0; index < model->mapping_count; ++index) {
        if (model->mappings[index].address != MAP_FAILED) {
            (void) munmap(model->mappings[index].address, model->mappings[index].size);
        }
    }
    free(model->mappings);
    free(model->tensors);
    free(model);
}

static jf_status map_package_file(
    int directory,
    const struct manifest_segment * descriptor,
    jf_model * model,
    size_t mapping_index,
    int verify_hash,
    int evict_before_open,
    void ** address_output,
    size_t * size_output
) {
    char path[256];
    memcpy(path, descriptor->path, descriptor->path_length);
    path[descriptor->path_length] = '\0';

    int file = openat(directory, path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file < 0) {
        return status_error(JF_STATUS_IO_ERROR, "open %s: %s", path, strerror(errno));
    }
    struct stat metadata;
    if (fstat(file, &metadata) != 0) {
        const int saved_errno = errno;
        close(file);
        return status_error(JF_STATUS_IO_ERROR, "stat %s: %s", path, strerror(saved_errno));
    }
    if (!S_ISREG(metadata.st_mode) || metadata.st_size <= 0 ||
        (uint64_t) metadata.st_size != descriptor->file_size || descriptor->file_size > SIZE_MAX) {
        close(file);
        return status_error(JF_STATUS_FORMAT_ERROR, "%s size or file type does not match manifest", path);
    }

#ifdef POSIX_FADV_DONTNEED
    if (evict_before_open) {
        const int advice_result = posix_fadvise(file, 0, 0, POSIX_FADV_DONTNEED);
        if (advice_result != 0) {
            close(file);
            return status_error(
                JF_STATUS_IO_ERROR,
                "evict %s from page cache: %s",
                path,
                strerror(advice_result)
            );
        }
    }
#else
    if (evict_before_open) {
        close(file);
        return status_error(JF_STATUS_IO_ERROR, "page-cache eviction is unsupported");
    }
#endif

    void * address = mmap(NULL, (size_t) metadata.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    const int saved_errno = errno;
    close(file);
    if (address == MAP_FAILED) {
        return status_error(JF_STATUS_IO_ERROR, "map %s: %s", path, strerror(saved_errno));
    }
    model->mappings[mapping_index] = (struct jf_mapping){address, (size_t) metadata.st_size};
    model->mapping_count = mapping_index + 1;

    if (verify_hash) {
        uint8_t digest[32];
        jf_sha256_buffer(address, (size_t) metadata.st_size, digest);
        if (memcmp(digest, descriptor->sha256, sizeof(digest)) != 0) {
            return status_error(JF_STATUS_FORMAT_ERROR, "%s checksum does not match manifest", path);
        }
    }

    *address_output = address;
    *size_output = (size_t) metadata.st_size;
    return status_ok();
}

jf_status jf_tensor_type_layout(
    uint32_t type,
    uint32_t * block_elements,
    uint32_t * block_bytes
) {
    if (block_elements == NULL || block_bytes == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "tensor layout outputs are required");
    }
    static const uint16_t elements[] = {
        0, 1, 1, 32, 32, 32, 32, 32, 32,
        256, 256, 256, 256, 256, 256, 256, 256, 256,
        256, 32, 256, 256, 256, 1, 1, 1, 1, 1, 256,
        1, 256, 256, 32, 64, 128, 64,
    };
    static const uint16_t bytes[] = {
        0, 4, 2, 18, 20, 22, 24, 34, 36,
        84, 110, 144, 176, 210, 292, 66, 74, 98,
        50, 18, 110, 82, 136, 1, 2, 4, 8, 8, 56,
        2, 54, 66, 17, 36, 18, 18,
    };
    if (type == 0 || type >= sizeof(elements) / sizeof(elements[0])) {
        return status_error(JF_STATUS_FORMAT_ERROR, "unsupported JFM tensor type %u", type);
    }
    *block_elements = elements[type];
    *block_bytes = bytes[type];
    return status_ok();
}

static jf_status map_tensor_segment(
    int directory,
    const struct manifest_segment * descriptor,
    jf_model * model,
    size_t mapping_index,
    size_t * tensor_index,
    int verify_hash,
    int evict_before_open
) {
    void * address = NULL;
    size_t file_size = 0;
    jf_status status = map_package_file(
        directory,
        descriptor,
        model,
        mapping_index,
        verify_hash,
        evict_before_open,
        &address,
        &file_size
    );
    if (status.code != JF_STATUS_OK) {
        return status;
    }
    if (file_size < JF_SEGMENT_HEADER_SIZE) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment is truncated");
    }

    const uint8_t * bytes = address;
    if (memcmp(bytes, JF_SEGMENT_MAGIC, sizeof(JF_SEGMENT_MAGIC)) != 0 ||
        jf_read_u32_le(bytes + 8) != JF_MODEL_FORMAT_VERSION ||
        !bytes_are_zero(bytes + 32, JF_SEGMENT_HEADER_SIZE - 32U)) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment has an unsupported header");
    }
    const uint32_t tensor_count = jf_read_u32_le(bytes + 12);
    const uint64_t index_bytes_u64 = jf_read_u64_le(bytes + 16);
    const uint64_t payload_bytes = jf_read_u64_le(bytes + 24);
    if (tensor_count != descriptor->tensor_count || payload_bytes != descriptor->tensor_bytes ||
        index_bytes_u64 > SIZE_MAX ||
        !range_fits(JF_SEGMENT_HEADER_SIZE, (size_t) index_bytes_u64, file_size)) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment has an invalid index");
    }

    const size_t index_end = JF_SEGMENT_HEADER_SIZE + (size_t) index_bytes_u64;
    if (index_end > SIZE_MAX - (JF_FORMAT_ALIGNMENT - 1U)) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment index overflows");
    }
    const size_t data_start =
        (index_end + JF_FORMAT_ALIGNMENT - 1U) / JF_FORMAT_ALIGNMENT * JF_FORMAT_ALIGNMENT;
    if (data_start > file_size) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment data starts past end of file");
    }

    size_t cursor = JF_SEGMENT_HEADER_SIZE;
    uint64_t previous_data_end = data_start;
    uint64_t tensor_bytes = 0;
    for (uint32_t index = 0; index < tensor_count; ++index) {
        if (!range_fits(cursor, JF_TENSOR_RECORD_SIZE, index_end)) {
            return status_error(JF_STATUS_FORMAT_ERROR, "tensor record is truncated");
        }
        const uint8_t * record = bytes + cursor;
        const uint32_t type = jf_read_u32_le(record);
        const uint32_t rank = jf_read_u32_le(record + 4);
        const int32_t layer = jf_read_i32_le(record + 8);
        const uint64_t data_offset = jf_read_u64_le(record + 48);
        const uint64_t data_length = jf_read_u64_le(record + 56);
        const uint32_t name_length = jf_read_u32_le(record + 64);
        const uint32_t block_elements = jf_read_u32_le(record + 68);
        const uint32_t block_bytes = jf_read_u32_le(record + 72);
        const uint32_t flags = jf_read_u32_le(record + 12);
        const uint32_t reserved = jf_read_u32_le(record + 76);
        cursor += JF_TENSOR_RECORD_SIZE;

        uint32_t canonical_block_elements = 0;
        uint32_t canonical_block_bytes = 0;
        if (jf_tensor_type_layout(
                type,
                &canonical_block_elements,
                &canonical_block_bytes
            ).code != JF_STATUS_OK ||
            block_elements != canonical_block_elements ||
            block_bytes != canonical_block_bytes || flags != 0 || reserved != 0 ||
            rank == 0 || rank > 4 ||
            name_length == 0 || name_length > 1024 ||
            block_elements == 0 || block_bytes == 0 ||
            !range_fits(cursor, name_length, index_end) ||
            data_offset > SIZE_MAX || data_length > SIZE_MAX ||
            data_offset % JF_FORMAT_ALIGNMENT != 0 || data_offset < previous_data_end ||
            !range_fits((size_t) data_offset, (size_t) data_length, file_size)) {
            return status_error(JF_STATUS_FORMAT_ERROR, "tensor segment contains an invalid record");
        }
        if ((descriptor->kind == JF_SEGMENT_LAYER && layer != descriptor->layer) ||
            (descriptor->kind != JF_SEGMENT_LAYER && layer != -1)) {
            return status_error(JF_STATUS_FORMAT_ERROR, "tensor layer does not match its segment");
        }
        if (*tensor_index >= model->tensor_count) {
            return status_error(JF_STATUS_FORMAT_ERROR, "segment declares too many tensors");
        }

        uint64_t row_count = 1;
        uint64_t row_elements = 0;
        for (uint32_t dimension = 0; dimension < 4; ++dimension) {
            const uint64_t extent = jf_read_u64_le(record + 16 + dimension * 8U);
            if (extent == 0 || (dimension >= rank && extent != 1)) {
                return status_error(JF_STATUS_FORMAT_ERROR, "tensor shape is invalid");
            }
            if (dimension == 0) {
                row_elements = extent;
            } else if (!multiply_u64(row_count, extent, &row_count)) {
                return status_error(JF_STATUS_FORMAT_ERROR, "tensor shape overflows");
            }
        }
        uint64_t expected_bytes = 0;
        uint64_t row_bytes = 0;
        if (row_elements % block_elements != 0 ||
            !multiply_u64(row_elements / block_elements, block_bytes, &row_bytes) ||
            !multiply_u64(row_bytes, row_count, &expected_bytes) ||
            expected_bytes != data_length ||
            !add_u64(data_offset, data_length, &previous_data_end) ||
            !add_u64(tensor_bytes, data_length, &tensor_bytes)) {
            return status_error(JF_STATUS_FORMAT_ERROR, "tensor storage size is inconsistent");
        }

        struct jf_tensor * tensor = &model->tensors[*tensor_index];
        tensor->name = (const char *) bytes + cursor;
        tensor->name_length = name_length;
        tensor->type = type;
        tensor->rank = rank;
        tensor->storage_block_elements = block_elements;
        tensor->storage_block_bytes = block_bytes;
        tensor->layer = layer;
        tensor->data = bytes + data_offset;
        tensor->size = data_length;
        for (uint32_t dimension = 0; dimension < 4; ++dimension) {
            tensor->shape[dimension] = jf_read_u64_le(record + 16 + dimension * 8U);
        }
        cursor += name_length;
        cursor = (cursor + 7U) & ~(size_t) 7U;
        ++*tensor_index;
    }
    if (cursor != index_end || tensor_bytes != descriptor->tensor_bytes) {
        return status_error(JF_STATUS_FORMAT_ERROR, "tensor index or payload total does not match manifest");
    }
#ifdef MADV_SEQUENTIAL
    (void) madvise(address, file_size, MADV_SEQUENTIAL);
#endif
    return status_ok();
}

static jf_status map_metadata_segment(
    int directory,
    const struct manifest_segment * descriptor,
    jf_model * model,
    size_t mapping_index,
    int verify_hash,
    int evict_before_open
) {
    void * address = NULL;
    size_t size = 0;
    jf_status status = map_package_file(
        directory,
        descriptor,
        model,
        mapping_index,
        verify_hash,
        evict_before_open,
        &address,
        &size
    );
    if (status.code != JF_STATUS_OK) {
        return status;
    }
    if (size < 4 || memcmp(address, "GGUF", 4) != 0) {
        return status_error(JF_STATUS_FORMAT_ERROR, "metadata segment is not a GGUF header");
    }
    model->gguf_metadata = address;
    model->gguf_metadata_size = size;
    return status_ok();
}

static int compare_tensor_names(const void * left_pointer, const void * right_pointer) {
    const struct jf_tensor * left = left_pointer;
    const struct jf_tensor * right = right_pointer;
    const size_t common = left->name_length < right->name_length
        ? left->name_length
        : right->name_length;
    const int result = memcmp(left->name, right->name, common);
    if (result != 0) return result;
    return (left->name_length > right->name_length) - (left->name_length < right->name_length);
}

static int sort_and_validate_tensor_names(jf_model * model) {
    qsort(model->tensors, model->tensor_count, sizeof(*model->tensors), compare_tensor_names);
    for (size_t index = 1; index < model->tensor_count; ++index) {
        if (compare_tensor_names(&model->tensors[index - 1], &model->tensors[index]) == 0) {
            return 0;
        }
    }
    return 1;
}

jf_status jf_model_open(
    const char * package_path,
    const jf_stage_plan * plan,
    jf_model ** output
) {
    if (package_path == NULL || package_path[0] == '\0' || plan == NULL || output == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "package path, plan, and output are required");
    }
    *output = NULL;

    int directory = open(package_path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory < 0) {
        return status_error(JF_STATUS_IO_ERROR, "open package %s: %s", package_path, strerror(errno));
    }

    uint8_t * manifest = NULL;
    size_t manifest_size = 0;
    jf_status status = read_manifest(directory, &manifest, &manifest_size);
    if (status.code != JF_STATUS_OK) {
        close(directory);
        return status;
    }
    if (manifest_size < JF_MANIFEST_HEADER_SIZE ||
        memcmp(manifest, JF_MANIFEST_MAGIC, sizeof(JF_MANIFEST_MAGIC)) != 0 ||
        jf_read_u32_le(manifest + 8) != JF_MODEL_FORMAT_VERSION ||
        jf_read_u32_le(manifest + 60) != 0) {
        status = status_error(JF_STATUS_FORMAT_ERROR, "manifest has an unsupported header");
        goto fail_header;
    }

    const uint32_t layer_count = jf_read_u32_le(manifest + 12);
    const uint64_t source_size = jf_read_u64_le(manifest + 16);
    const uint32_t segment_count = jf_read_u32_le(manifest + 56);
    if (layer_count == 0 || layer_count > JF_MAX_LAYERS || source_size == 0 ||
        plan->layer_start >= plan->layer_end || plan->layer_end > layer_count) {
        status = status_error(JF_STATUS_INVALID_ARGUMENT, "stage range or model dimensions are invalid");
        goto fail_header;
    }
    if (segment_count < layer_count + 3U || segment_count > layer_count + 4U ||
        segment_count > JF_MAX_SEGMENTS) {
        status = status_error(JF_STATUS_FORMAT_ERROR, "manifest segment topology is invalid");
        goto fail_header;
    }

    struct manifest_segment * segments = calloc(segment_count, sizeof(*segments));
    uint8_t * layers_seen = calloc(layer_count, sizeof(*layers_seen));
    if (segments == NULL || layers_seen == NULL) {
        free(segments);
        free(layers_seen);
        status = status_error(JF_STATUS_OUT_OF_MEMORY, "allocate manifest topology index");
        goto fail_header;
    }

    size_t cursor = JF_MANIFEST_HEADER_SIZE;
    uint64_t selected_tensors = 0;
    uint64_t total_tensors = 0;
    uint64_t selected_weights = 0;
    uint64_t total_weights = 0;
    uint64_t selected_mapped = 0;
    size_t selected_segments = 0;
    uint32_t metadata_count = 0;
    uint32_t shared_count = 0;
    uint32_t input_count = 0;
    uint32_t output_count = 0;
    for (uint32_t index = 0; index < segment_count; ++index) {
        if (!range_fits(cursor, JF_MANIFEST_SEGMENT_SIZE, manifest_size)) {
            status = status_error(JF_STATUS_FORMAT_ERROR, "manifest segment index is truncated");
            goto fail_manifest;
        }
        const uint8_t * record = manifest + cursor;
        struct manifest_segment * segment = &segments[index];
        segment->kind = jf_read_u32_le(record);
        segment->layer = jf_read_i32_le(record + 4);
        segment->tensor_count = jf_read_u64_le(record + 8);
        segment->tensor_bytes = jf_read_u64_le(record + 16);
        segment->file_size = jf_read_u64_le(record + 24);
        memcpy(segment->sha256, record + 32, sizeof(segment->sha256));
        segment->path_length = jf_read_u32_le(record + 64);
        const uint32_t reserved = jf_read_u32_le(record + 68);
        cursor += JF_MANIFEST_SEGMENT_SIZE;
        if (!range_fits(cursor, segment->path_length, manifest_size) ||
            !safe_segment_path((const char *) manifest + cursor, segment->path_length) ||
            segment->tensor_count > SIZE_MAX || segment->file_size == 0 ||
            segment->file_size > SIZE_MAX || segment->kind > JF_SEGMENT_METADATA ||
            reserved != 0 ||
            (segment->kind != JF_SEGMENT_METADATA &&
             (segment->tensor_count == 0 || segment->tensor_bytes == 0))) {
            status = status_error(JF_STATUS_FORMAT_ERROR, "manifest contains an invalid segment");
            goto fail_manifest;
        }
        segment->path = (const char *) manifest + cursor;
        cursor += segment->path_length;
        if (segment->kind == JF_SEGMENT_LAYER) {
            if (segment->layer < 0 || (uint32_t) segment->layer >= layer_count ||
                layers_seen[segment->layer] != 0 || segment->tensor_count == 0) {
                status = status_error(JF_STATUS_FORMAT_ERROR, "manifest layer topology is invalid");
                goto fail_manifest;
            }
            layers_seen[segment->layer] = 1;
        } else {
            if (segment->layer != -1) {
                status = status_error(JF_STATUS_FORMAT_ERROR, "global segment declares a layer");
                goto fail_manifest;
            }
            switch (segment->kind) {
            case JF_SEGMENT_SHARED: ++shared_count; break;
            case JF_SEGMENT_INPUT: ++input_count; break;
            case JF_SEGMENT_OUTPUT: ++output_count; break;
            case JF_SEGMENT_METADATA:
                ++metadata_count;
                if (segment->tensor_count != 0 || segment->tensor_bytes != 0) {
                    status = status_error(JF_STATUS_FORMAT_ERROR, "metadata segment declares tensors");
                    goto fail_manifest;
                }
                break;
            case JF_SEGMENT_LAYER: break;
            }
        }

        if (!add_u64(total_tensors, segment->tensor_count, &total_tensors) ||
            !add_u64(total_weights, segment->tensor_bytes, &total_weights) ||
            total_tensors > JF_MAX_TENSORS) {
            status = status_error(JF_STATUS_FORMAT_ERROR, "manifest tensor totals overflow");
            goto fail_manifest;
        }
        const int selected = segment_selected(segment, *plan) ||
            (segment->kind == JF_SEGMENT_OUTPUT && plan->layer_end == layer_count);
        if (selected) {
            ++selected_segments;
            if (!add_u64(selected_tensors, segment->tensor_count, &selected_tensors) ||
                !add_u64(selected_weights, segment->tensor_bytes, &selected_weights) ||
                !add_u64(selected_mapped, segment->file_size, &selected_mapped)) {
                status = status_error(JF_STATUS_FORMAT_ERROR, "selected model totals overflow");
                goto fail_manifest;
            }
        }
    }
    if (cursor != manifest_size || metadata_count != 1 || shared_count > 1 ||
        input_count != 1 || output_count != 1) {
        status = status_error(JF_STATUS_FORMAT_ERROR, "manifest does not define one complete model");
        goto fail_manifest;
    }
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (layers_seen[layer] == 0) {
            status = status_error(JF_STATUS_FORMAT_ERROR, "manifest is missing a model layer");
            goto fail_manifest;
        }
    }
    qsort(segments, segment_count, sizeof(*segments), compare_segment_paths);
    for (uint32_t index = 1; index < segment_count; ++index) {
        if (same_path(&segments[index - 1], &segments[index])) {
            status = status_error(JF_STATUS_FORMAT_ERROR, "manifest contains duplicate segment paths");
            goto fail_manifest;
        }
    }
    if (selected_tensors > SIZE_MAX || selected_mapped > SIZE_MAX) {
        status = status_error(JF_STATUS_FORMAT_ERROR, "selected model exceeds addressable memory");
        goto fail_manifest;
    }

    jf_model * model = calloc(1, sizeof(*model));
    if (model == NULL) {
        status = status_error(JF_STATUS_OUT_OF_MEMORY, "allocate model");
        goto fail_manifest;
    }
    model->mappings = calloc(selected_segments, sizeof(*model->mappings));
    model->tensors = calloc((size_t) selected_tensors, sizeof(*model->tensors));
    model->tensor_count = (size_t) selected_tensors;
    if ((selected_segments > 0 && model->mappings == NULL) ||
        (selected_tensors > 0 && model->tensors == NULL)) {
        close_model_contents(model);
        status = status_error(JF_STATUS_OUT_OF_MEMORY, "allocate selected model index");
        goto fail_manifest;
    }
    memcpy(model->source_sha256, manifest + 24, sizeof(model->source_sha256));
    model->stats = (jf_model_stats){
        .layer_start = plan->layer_start,
        .layer_end = plan->layer_end,
        .layer_count = layer_count,
        .selected_weight_bytes = selected_weights,
        .total_weight_bytes = total_weights,
        .mapped_bytes = selected_mapped,
        .selected_tensor_count = selected_tensors,
        .total_tensor_count = total_tensors,
    };

    size_t mapping_index = 0;
    size_t tensor_index = 0;
    for (uint32_t index = 0; index < segment_count; ++index) {
        const int selected = segment_selected(&segments[index], *plan) ||
            (segments[index].kind == JF_SEGMENT_OUTPUT && plan->layer_end == layer_count);
        if (!selected) {
            continue;
        }
        status = segments[index].kind == JF_SEGMENT_METADATA
            ? map_metadata_segment(
                directory,
                &segments[index],
                model,
                mapping_index,
                plan->verify_hashes != 0,
                plan->evict_before_open != 0
            )
            : map_tensor_segment(
                directory,
                &segments[index],
                model,
                mapping_index,
                &tensor_index,
                plan->verify_hashes != 0,
                plan->evict_before_open != 0
            );
        if (status.code != JF_STATUS_OK) {
            close_model_contents(model);
            goto fail_manifest;
        }
        ++mapping_index;
    }
    if (tensor_index != model->tensor_count || !sort_and_validate_tensor_names(model)) {
        close_model_contents(model);
        status = status_error(JF_STATUS_FORMAT_ERROR, "selected tensor index is incomplete or ambiguous");
        goto fail_manifest;
    }

    free(layers_seen);
    free(segments);
    free(manifest);
    close(directory);
    *output = model;
    return status_ok();

fail_manifest:
    free(layers_seen);
    free(segments);
fail_header:
    free(manifest);
    close(directory);
    return status;
}

void jf_model_close(jf_model * model) {
    close_model_contents(model);
}

jf_model_stats jf_model_get_stats(const jf_model * model) {
    return model == NULL ? (jf_model_stats){0} : model->stats;
}

size_t jf_model_tensor_count(const jf_model * model) {
    return model == NULL ? 0 : model->tensor_count;
}

static void copy_tensor_view(const struct jf_tensor * source, jf_tensor_view * destination) {
    destination->name = source->name;
    destination->name_length = source->name_length;
    destination->type = source->type;
    destination->rank = source->rank;
    destination->storage_block_elements = source->storage_block_elements;
    destination->storage_block_bytes = source->storage_block_bytes;
    destination->layer = source->layer;
    destination->data = source->data;
    destination->size = source->size;
    memcpy(destination->shape, source->shape, sizeof(destination->shape));
}

jf_status jf_model_tensor_at(
    const jf_model * model,
    size_t index,
    jf_tensor_view * tensor
) {
    if (model == NULL || tensor == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "model and tensor output are required");
    }
    if (index >= model->tensor_count) {
        return status_error(JF_STATUS_NOT_FOUND, "tensor index %zu is out of range", index);
    }
    copy_tensor_view(&model->tensors[index], tensor);
    return status_ok();
}

jf_status jf_model_find_tensor(
    const jf_model * model,
    const char * name,
    jf_tensor_view * tensor
) {
    if (model == NULL || name == NULL || name[0] == '\0' || tensor == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "model, tensor name, and output are required");
    }
    const size_t length = strlen(name);
    for (size_t index = 0; index < model->tensor_count; ++index) {
        const struct jf_tensor * candidate = &model->tensors[index];
        if (candidate->name_length == length && memcmp(candidate->name, name, length) == 0) {
            copy_tensor_view(candidate, tensor);
            return status_ok();
        }
    }
    return status_error(JF_STATUS_NOT_FOUND, "tensor %s is not selected", name);
}

jf_status jf_model_get_source_sha256(const jf_model * model, uint8_t digest[32]) {
    if (model == NULL || digest == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "model and digest output are required");
    }
    memcpy(digest, model->source_sha256, 32);
    return status_ok();
}

jf_status jf_model_get_gguf_metadata(
    const jf_model * model,
    const void ** data,
    size_t * size
) {
    if (model == NULL || data == NULL || size == NULL) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "model, metadata data, and size are required");
    }
    *data = model->gguf_metadata;
    *size = model->gguf_metadata_size;
    return status_ok();
}

struct prefetch_context {
    jf_model * model;
    size_t page_size;
    atomic_size_t next_mapping;
    uint64_t * mapping_checksums;
};

static void * prefetch_worker(void * argument) {
    struct prefetch_context * context = argument;
    for (;;) {
        const size_t mapping = atomic_fetch_add(&context->next_mapping, 1);
        if (mapping >= context->model->mapping_count) {
            return NULL;
        }
        const struct jf_mapping * region = &context->model->mappings[mapping];
#ifdef MADV_WILLNEED
        (void) madvise(region->address, region->size, MADV_WILLNEED);
#endif
        const volatile uint8_t * bytes = region->address;
        uint64_t value = 1469598103934665603ULL;
        for (size_t offset = 0; offset < region->size; offset += context->page_size) {
            value = (value ^ bytes[offset]) * 1099511628211ULL;
        }
        value = (value ^ bytes[region->size - 1]) * 1099511628211ULL;
        context->mapping_checksums[mapping] = value;
    }
}

jf_status jf_model_prefetch_parallel(
    jf_model * model,
    uint32_t thread_count,
    uint64_t * checksum
) {
    if (model == NULL || thread_count == 0 || thread_count > 256) {
        return status_error(JF_STATUS_INVALID_ARGUMENT, "model and 1-256 prefetch threads are required");
    }
    const long page_size_result = sysconf(_SC_PAGESIZE);
    if (page_size_result <= 0) {
        return status_error(JF_STATUS_IO_ERROR, "query system page size: %s", strerror(errno));
    }
    const size_t page_size = (size_t) page_size_result;
    uint64_t * mapping_checksums = calloc(model->mapping_count, sizeof(*mapping_checksums));
    if (model->mapping_count > 0 && mapping_checksums == NULL) {
        return status_error(JF_STATUS_OUT_OF_MEMORY, "allocate prefetch checksums");
    }
    struct prefetch_context context = {
        .model = model,
        .page_size = page_size,
        .next_mapping = 0,
        .mapping_checksums = mapping_checksums,
    };
    const size_t workers = model->mapping_count < thread_count
        ? model->mapping_count
        : thread_count;
    pthread_t * threads = workers > 1 ? calloc(workers - 1, sizeof(*threads)) : NULL;
    if (workers > 1 && threads == NULL) {
        free(mapping_checksums);
        return status_error(JF_STATUS_OUT_OF_MEMORY, "allocate prefetch workers");
    }
    size_t started = 0;
    for (; started + 1 < workers; ++started) {
        const int result = pthread_create(&threads[started], NULL, prefetch_worker, &context);
        if (result != 0) {
            break;
        }
    }
    (void) prefetch_worker(&context);
    for (size_t index = 0; index < started; ++index) {
        (void) pthread_join(threads[index], NULL);
    }
    free(threads);
    if (started + 1 < workers) {
        free(mapping_checksums);
        return status_error(JF_STATUS_IO_ERROR, "create prefetch worker thread");
    }

    uint64_t value = 1469598103934665603ULL;
    for (size_t mapping = 0; mapping < model->mapping_count; ++mapping) {
        value = (value ^ mapping_checksums[mapping]) * 1099511628211ULL;
    }
    free(mapping_checksums);
    if (checksum != NULL) {
        *checksum = value;
    }
    return status_ok();
}

jf_status jf_model_prefetch(jf_model * model, uint64_t * checksum) {
    return jf_model_prefetch_parallel(model, 1, checksum);
}

const char * jf_status_code_name(jf_status_code code) {
    switch (code) {
    case JF_STATUS_OK: return "ok";
    case JF_STATUS_INVALID_ARGUMENT: return "invalid_argument";
    case JF_STATUS_IO_ERROR: return "io_error";
    case JF_STATUS_FORMAT_ERROR: return "format_error";
    case JF_STATUS_NOT_FOUND: return "not_found";
    case JF_STATUS_OUT_OF_MEMORY: return "out_of_memory";
    }
    return "unknown";
}
