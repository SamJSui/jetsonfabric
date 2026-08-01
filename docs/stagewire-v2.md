# Stagewire v2

`stagewire` is JetsonFabric's versioned contract for one stage operation. It carries metadata plus raw payload bytes between logical nodes and their runtime workers. Tensor payloads are never base64-encoded or represented as JSON arrays.

## Media type

```text
application/vnd.jetsonfabric.stage.v2+octet-stream
```

## Frame layout

All integer fields in the fixed header use network byte order.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic bytes `JFST` |
| 4 | 2 | protocol version, currently `2` |
| 6 | 2 | flags, currently `0` |
| 8 | 4 | metadata JSON length |
| 12 | 8 | raw payload length |
| 20 | variable | UTF-8 JSON metadata |
| after metadata | variable | raw payload bytes |

A request or response body contains exactly one frame. Unsupported versions, oversized metadata or payloads, truncation, trailing bytes, shape mismatches, and checksum mismatches are rejected.

## Metadata

Stage metadata includes:

- protocol version;
- operation: execute, close session, or rollback session;
- session, request, model, and node identity;
- optional managed deployment ID, positive epoch, and 64-character model
  SHA-256, which must be present as a complete set;
- inference phase and decode step;
- stage index/count and assigned layer range;
- payload kind;
- text encoding or tensor dtype/shape/byte order/layout;
- payload byte length and CRC32;
- transport identifier;
- request limits, rollback count, prompt token IDs, token batches, byte counts,
  stage timings, verification width, and optional error details.

Stage position remains count-based. There is no first, intermediate, or final role string on the wire.

## Transport authentication

Node facades and runtime workers require `X-JetsonFabric-Cluster-Token` on
Stagewire writes. Go diagnostic clients and the C++ runtime peer client attach
the configured `JETSONFABRIC_CLUSTER_TOKEN`. Relay mode sends traffic through
the peer node facade; direct mode sends it to the peer runtime after deployment
admission verifies the advertised endpoint. This is shared-secret
authentication, not transport security: Stagewire remains plaintext HTTP until
TLS and per-node identity are implemented.

## Payloads

Supported semantic payload kinds are:

- `text`: UTF-8 prompt bytes;
- `tokens`: typed token-ID bytes;
- `activation`: typed hidden-state tensor bytes;
- `sampled_token`: one typed token ID;
- `sampled_tokens`: a target-verified token batch.

Tensor payloads require:

```text
byte_order = little
layout     = row_major
```

Supported v2 dtype labels are `u8`, `i8`, `f16`, `bf16`, `i32`, `u32`, `f32`, `i64`, `u64`, and `f64`. The product of shape dimensions and dtype width must match the payload length exactly.

The llama.cpp pipeline supports F32 and F16 activations with shape
`[sequence_length, hidden_size]` and little-endian 32-bit sampled tokens.

## Ownership

```text
internal/inference
  defines legal semantic transitions

internal/stagewire
  encodes, decodes, versions, validates, and checksums Go frames

internal/stageexec
  sends stage outputs for the diagnostic layer-split API

internal/runtimebridge
  streams frames between a node API and its local runtime

runtime/protocol
  implements the matching C++ frame contract

runtime/pipeline_parallel + runtime/transport
  validate and forward generation stage outputs from the stage-0 runtime
```

## Current validation

Two complementary tests exercise the same wire contract:

1. The synthetic integration creates a deterministic `f32[4,16]` activation, sends it through two logical nodes and runtimes, and returns the activation CRC32 as a sampled token.
2. The real-model integration sends llama.cpp hidden activations between assigned layer ranges during prefill and decode, verifies byte and CRC continuity, requires authenticated peer calls, and requires the runtime-owned greedy token stream to match a one-runtime baseline.

These tests prove binary activation transport, real partial-layer execution,
partitioned stage-local weight residency, target-verified token batches, and
distributed KV-cache rollback. Physical two-Jetson CUDA validation separately
proves both relay and direct runtime transport; performance remains workload
dependent.
