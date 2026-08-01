#include "sha256.h"

#include <string.h>

static const uint32_t round_constants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

static uint32_t rotate_right(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32U - count));
}

static uint32_t read_u32_be(const uint8_t * bytes) {
    return ((uint32_t) bytes[0] << 24U) |
        ((uint32_t) bytes[1] << 16U) |
        ((uint32_t) bytes[2] << 8U) |
        (uint32_t) bytes[3];
}

static void write_u32_be(uint8_t * bytes, uint32_t value) {
    bytes[0] = (uint8_t) (value >> 24U);
    bytes[1] = (uint8_t) (value >> 16U);
    bytes[2] = (uint8_t) (value >> 8U);
    bytes[3] = (uint8_t) value;
}

static void transform(jf_sha256_context * context, const uint8_t block[64]) {
    uint32_t words[64];
    for (unsigned index = 0; index < 16; ++index) {
        words[index] = read_u32_be(block + index * 4U);
    }
    for (unsigned index = 16; index < 64; ++index) {
        const uint32_t s0 = rotate_right(words[index - 15], 7) ^
            rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3U);
        const uint32_t s1 = rotate_right(words[index - 2], 17) ^
            rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];

    for (unsigned index = 0; index < 64; ++index) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t temporary1 = h + sum1 + choose + round_constants[index] + words[index];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void jf_sha256_init(jf_sha256_context * context) {
    *context = (jf_sha256_context){
        .state = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
        },
    };
}

void jf_sha256_update(jf_sha256_context * context, const void * data, size_t size) {
    const uint8_t * bytes = data;
    context->bit_count += (uint64_t) size * 8U;
    while (size > 0) {
        const size_t available = sizeof(context->buffer) - context->buffer_size;
        const size_t count = size < available ? size : available;
        memcpy(context->buffer + context->buffer_size, bytes, count);
        context->buffer_size += count;
        bytes += count;
        size -= count;
        if (context->buffer_size == sizeof(context->buffer)) {
            transform(context, context->buffer);
            context->buffer_size = 0;
        }
    }
}

void jf_sha256_final(jf_sha256_context * context, uint8_t digest[32]) {
    const uint64_t original_bit_count = context->bit_count;
    const uint8_t marker = 0x80U;
    jf_sha256_update(context, &marker, 1);
    const uint8_t zero = 0;
    while (context->buffer_size != 56) {
        jf_sha256_update(context, &zero, 1);
    }
    uint8_t length[8];
    for (unsigned index = 0; index < 8; ++index) {
        length[7U - index] = (uint8_t) (original_bit_count >> (index * 8U));
    }
    jf_sha256_update(context, length, sizeof(length));
    for (unsigned index = 0; index < 8; ++index) {
        write_u32_be(digest + index * 4U, context->state[index]);
    }
    memset(context, 0, sizeof(*context));
}

void jf_sha256_buffer(const void * data, size_t size, uint8_t digest[32]) {
    jf_sha256_context context;
    jf_sha256_init(&context);
    jf_sha256_update(&context, data, size);
    jf_sha256_final(&context, digest);
}
