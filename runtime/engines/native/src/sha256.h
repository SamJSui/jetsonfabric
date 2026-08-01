#ifndef JETSONFABRIC_NATIVE_SHA256_H
#define JETSONFABRIC_NATIVE_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jf_sha256_context {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t buffer[64];
    size_t buffer_size;
} jf_sha256_context;

void jf_sha256_init(jf_sha256_context * context);
void jf_sha256_update(jf_sha256_context * context, const void * data, size_t size);
void jf_sha256_final(jf_sha256_context * context, uint8_t digest[32]);
void jf_sha256_buffer(const void * data, size_t size, uint8_t digest[32]);

#ifdef __cplusplus
}
#endif

#endif
