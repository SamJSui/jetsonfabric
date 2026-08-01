#ifndef JETSONFABRIC_NATIVE_FORMAT_H
#define JETSONFABRIC_NATIVE_FORMAT_H

#include <stdint.h>

#define JF_MODEL_FORMAT_VERSION 2U
#define JF_MANIFEST_HEADER_SIZE 64U
#define JF_MANIFEST_SEGMENT_SIZE 72U
#define JF_SEGMENT_HEADER_SIZE 64U
#define JF_TENSOR_RECORD_SIZE 80U
#define JF_FORMAT_ALIGNMENT 64U

static const uint8_t JF_MANIFEST_MAGIC[8] = {'J', 'F', 'M', 'O', 'D', 'E', 'L', '2'};
static const uint8_t JF_SEGMENT_MAGIC[8] = {'J', 'F', 'S', 'E', 'G', 'M', '0', '2'};

enum jf_segment_kind {
    JF_SEGMENT_SHARED = 0,
    JF_SEGMENT_INPUT = 1,
    JF_SEGMENT_LAYER = 2,
    JF_SEGMENT_OUTPUT = 3,
    JF_SEGMENT_METADATA = 4
};

static inline uint32_t jf_read_u32_le(const uint8_t * data) {
    return ((uint32_t) data[0]) |
        ((uint32_t) data[1] << 8U) |
        ((uint32_t) data[2] << 16U) |
        ((uint32_t) data[3] << 24U);
}

static inline int32_t jf_read_i32_le(const uint8_t * data) {
    return (int32_t) jf_read_u32_le(data);
}

static inline uint64_t jf_read_u64_le(const uint8_t * data) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= (uint64_t) data[index] << (index * 8U);
    }
    return value;
}

#endif
