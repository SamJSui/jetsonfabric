# Stagewire v1

`stagewire` is JetsonFabric's versioned contract for one stage operation. It carries metadata plus raw payload bytes between logical nodes and their runtime workers. Tensor payloads are never base64-encoded or represented as JSON arrays.

## Media type

```text
application/vnd.jetsonfabric.stage.v1+octet-stream
```

## Frame layout

All integer fields in the fixed header use network byte order.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic bytes `JFST` |
| 4 | 2 | protocol version, currently `1` |
| 6 | 2 | flags, currently `0` |
| 8 | 4 | metadata JSON length |
| 12 | 8 | raw payload length |
| 20 | variable | UTF-8 JSON metadata |
| after metadata | variable | raw payload bytes |

A request or response body contains exactly one frame. Unsupported versions, oversized metadata or payloads, truncation, trailing bytes, shape mismatches, and checksum mismatches are rejected.

## Metadata

Stage metadata includes:

- protocol version;
- session, request, model, and node identity;
- inference phase and decode step;
- stage index/count and assigned layer range;
- payload kind;
- text encoding or tensor dtype/shape/byte order/layout;
- payload byte length and CRC32;
- transport identifier;
- request limits, token counts, byte counts, latency, and optional error details.

Stage position remains count-based. There is no first, intermediate, or final role string on the wire.

## Payloads

Supported semantic payload kinds are:

- `text`: UTF-8 prompt bytes;
- `tokens`: typed token-ID bytes;
- `activation`: typed hidden-state tensor bytes;
- `sampled_token`: one typed token ID.

Tensor payloads require:

```text
byte_order = little
layout     = row_major
```

Supported v1 dtype labels are `u8`, `i8`, `f16`, `bf16`, `i32`, `u32`, `f32`, `i64`, `u64`, and `f64`. The product of shape dimensions and dtype width must match the payload length exactly.

The current llama.cpp pipeline uses F32 activations with shape `[sequence_length, hidden_size]` and one little-endian 32-bit sampled token.

## Ownership

```text
internal/inference
  defines legal semantic transitions

internal/stagewire
  encodes, decodes, versions, validates, and checksums Go frames

internal/stageexec
  sends one stage output as the next stage input

internal/runtimebridge
  streams frames between a node API and its local runtime

runtime/protocol
  implements the matching C++ frame contract
```

## Current validation

Two complementary tests exercise the same wire contract:

1. The synthetic integration creates a deterministic `f32[4,16]` activation, sends it through two logical nodes and runtimes, and returns the activation CRC32 as a sampled token.
2. The real-model integration sends llama.cpp hidden activations between assigned layer ranges during prefill and decode, verifies byte and CRC continuity, and requires distributed greedy tokens to match a one-runtime baseline.

These tests prove binary activation transport and real partial-layer execution. They do not yet prove physical CUDA transport, direct runtime-to-runtime networking, or reduced per-node model-weight memory.
