# Stagewire v1

`stagewire` is JetsonFabric's versioned contract for one stage operation. It carries
metadata plus raw payload bytes between logical nodes and their runtime workers.
Tensor payloads are never base64-encoded and are never represented as JSON arrays.

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

A request or response body contains exactly one frame. Trailing bytes, truncated
payloads, unsupported versions, oversized frames, and checksum mismatches are
rejected.

## Metadata

Stage metadata includes:

- protocol version;
- session, request, and model identity;
- inference phase and decode step;
- stage index/count and assigned layer range;
- payload kind;
- text encoding or tensor dtype/shape/byte order/layout;
- payload byte length and CRC32;
- transport identifier;
- request limits, token counts, byte counts, latency, and optional error details.

Stage position remains count-based. There is no first/middle/final role string.

## Payloads

Supported semantic payload kinds are:

- `text`: UTF-8 bytes;
- `tokens`: typed token-ID bytes;
- `activation`: typed hidden-state tensor bytes;
- `sampled_token`: one typed token ID.

Tensor payloads currently require:

```text
byte_order = little
layout     = row_major
```

Supported v1 dtype labels are `u8`, `i8`, `f16`, `bf16`, `i32`, `u32`, `f32`,
`i64`, `u64`, and `f64`. The product of shape dimensions and dtype width must
match the payload length exactly.

## Ownership

```text
internal/inference
  defines legal semantic transitions

internal/stagewire
  encodes, decodes, versions, validates, and checksums frames

internal/stageexec
  sends one stage output as the next stage input

internal/runtimebridge
  streams frames between the node API and local runtime

runtime/protocol
  implements the matching C++ frame contract
```

## Current validation milestone

The CI synthetic engine produces a deterministic `f32[4,16]` activation on stage
0. The activation crosses two real logical node APIs and two supervised C++
runtime workers. Stage 1 receives the activation and returns its CRC32 as a
`u32[1]` sampled-token payload. CI asserts that the outgoing and incoming
activation checksums and byte lengths match.

This proves JetsonFabric's activation data plane. It does not claim that llama.cpp
partial-layer execution or distributed KV-cache ownership is implemented yet.
