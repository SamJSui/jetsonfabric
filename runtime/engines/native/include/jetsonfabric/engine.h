#ifndef JETSONFABRIC_ENGINE_H
#define JETSONFABRIC_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jf_model jf_model;

typedef enum jf_tensor_type {
    JF_TENSOR_F32 = 1,
    JF_TENSOR_F16 = 2,
    JF_TENSOR_Q4_0 = 3,
    JF_TENSOR_Q4_1 = 4,
    JF_TENSOR_Q5_0 = 5,
    JF_TENSOR_Q5_1 = 6,
    JF_TENSOR_Q8_0 = 7,
    JF_TENSOR_Q8_1 = 8,
    JF_TENSOR_Q2_K = 9,
    JF_TENSOR_Q3_K = 10,
    JF_TENSOR_Q4_K = 11,
    JF_TENSOR_Q5_K = 12,
    JF_TENSOR_Q6_K = 13,
    JF_TENSOR_Q8_K = 14,
    JF_TENSOR_IQ2_XXS = 15,
    JF_TENSOR_IQ2_XS = 16,
    JF_TENSOR_IQ3_XXS = 17,
    JF_TENSOR_IQ1_S = 18,
    JF_TENSOR_IQ4_NL = 19,
    JF_TENSOR_IQ3_S = 20,
    JF_TENSOR_IQ2_S = 21,
    JF_TENSOR_IQ4_XS = 22,
    JF_TENSOR_I8 = 23,
    JF_TENSOR_I16 = 24,
    JF_TENSOR_I32 = 25,
    JF_TENSOR_I64 = 26,
    JF_TENSOR_F64 = 27,
    JF_TENSOR_IQ1_M = 28,
    JF_TENSOR_BF16 = 29,
    JF_TENSOR_TQ1_0 = 30,
    JF_TENSOR_TQ2_0 = 31,
    JF_TENSOR_MXFP4 = 32,
    JF_TENSOR_NVFP4 = 33,
    JF_TENSOR_Q1_0 = 34,
    JF_TENSOR_Q2_0 = 35
} jf_tensor_type;

typedef enum jf_status_code {
    JF_STATUS_OK = 0,
    JF_STATUS_INVALID_ARGUMENT = 1,
    JF_STATUS_IO_ERROR = 2,
    JF_STATUS_FORMAT_ERROR = 3,
    JF_STATUS_NOT_FOUND = 4,
    JF_STATUS_OUT_OF_MEMORY = 5
} jf_status_code;

typedef struct jf_status {
    jf_status_code code;
    char message[256];
} jf_status;

typedef struct jf_stage_plan {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t verify_hashes;
    uint32_t evict_before_open;
} jf_stage_plan;

typedef struct jf_model_stats {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t layer_count;
    uint64_t selected_weight_bytes;
    uint64_t total_weight_bytes;
    uint64_t mapped_bytes;
    uint64_t selected_tensor_count;
    uint64_t total_tensor_count;
} jf_model_stats;

typedef struct jf_tensor_view {
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
} jf_tensor_view;

// Returns the canonical number of logical elements and encoded bytes in one
// storage block. This is part of the JFM compatibility contract.
jf_status jf_tensor_type_layout(
    uint32_t type,
    uint32_t * block_elements,
    uint32_t * block_bytes
);

// Opens only the package segments needed by the requested half-open layer range.
jf_status jf_model_open(
    const char * package_path,
    const jf_stage_plan * plan,
    jf_model ** model
);

void jf_model_close(jf_model * model);

jf_model_stats jf_model_get_stats(const jf_model * model);

size_t jf_model_tensor_count(const jf_model * model);

jf_status jf_model_tensor_at(
    const jf_model * model,
    size_t index,
    jf_tensor_view * tensor
);

jf_status jf_model_find_tensor(
    const jf_model * model,
    const char * name,
    jf_tensor_view * tensor
);

// Faults mapped weight pages into memory. The checksum prevents the scan from
// being optimized away and is useful for reproducible loader benchmarks.
jf_status jf_model_prefetch(jf_model * model, uint64_t * checksum);

jf_status jf_model_prefetch_parallel(
    jf_model * model,
    uint32_t thread_count,
    uint64_t * checksum
);

jf_status jf_model_get_source_sha256(
    const jf_model * model,
    uint8_t digest[32]
);

jf_status jf_model_get_gguf_metadata(
    const jf_model * model,
    const void ** data,
    size_t * size
);

const char * jf_status_code_name(jf_status_code code);

#ifdef __cplusplus
}
#endif

#endif
