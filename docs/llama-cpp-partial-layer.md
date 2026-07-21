# llama.cpp partial-layer execution

JetsonFabric's first real pipeline-parallel engine path is pinned to llama.cpp
commit `bf2c86ddc0685f580595954056c2e77ebabfab4f`.

## Feasibility result

The pinned llama.cpp revision already supports two necessary primitives:

- raw F32 model input through `llama_batch.embd`;
- extracting hidden-state embeddings from a context.

It does not expose a public API that executes an arbitrary transformer layer
range. JetsonFabric therefore carries a narrow pinned patch in
`runtime/patches/llama_cpp_stage_range.patch`.

The patch:

- adds a model-load layer range `[layer_start, layer_end)`;
- creates only the input embedding, assigned transformer tensors, and final
  output tensors required by that range;
- adds a context layer range `[layer_start, layer_end)`;
- limits Llama and Qwen2 graph construction to that range;
- returns the hidden state instead of logits for non-final ranges;
- accepts raw hidden-state input for non-first ranges;
- preserves the existing final norm, output projection, and logits behavior for
  the final range.

All patched llama.cpp calls are contained under `runtime/adapters`. Stagewire,
Go orchestration, and the engine-neutral runtime interface do not expose ggml or
llama.cpp graph objects.

## Runtime path

```text
first stage
  text or tokens
  -> token embeddings
  -> assigned transformer layers
  -> f32[sequence_length, hidden_size] activation

intermediate stage
  activation
  -> assigned transformer layers
  -> activation

final stage
  activation
  -> assigned transformer layers
  -> final norm and output projection
  -> logits kept local
  -> greedy sampled token
```

Each stage owns a persistent llama.cpp context keyed by `session_id`. Prefill
creates the context. Decode reuses it and requires a monotonic `decode_step`.
Successful operations refresh the session's last-used timestamp. Explicit
cleanup releases all stage contexts, and each runtime independently reaps
sessions that remain idle for five minutes.

## Current contract

Activations use:

```text
dtype      = f32
shape      = [sequence_length, hidden_size]
byte_order = little
layout     = row_major
```

Sampled tokens use one little-endian `i32` element.

The native test compares:

1. full-model greedy generation;
2. exact full-model and per-stage resident tensor bytes;
3. independent stage-scoped model loads whose bytes are each below the full
   model and whose combined tensor payload reconstructs it;
4. split prefill through two stage adapters;
5. one split decode step using persistent stage contexts;
6. explicit session cleanup;
7. idle-session TTL cleanup.

The first and second sampled tokens must match the full-model baseline.

The two-node CPU E2E additionally requires the real activation to cross the
existing Stagewire path byte-for-byte before the final runtime samples. The
coordinator owns the multi-token decode loop and gives each stage operation a
unique request ID while retaining one server-generated session ID.

## Residency lifecycle

The GGUF artifact remains a registered file on local storage. Loading a
deployment creates only the tensors needed by its stage and reports them from
`GET /v1/deployment` under `model_memory`:

```text
layer_start, layer_end, layer_count
resident_weight_bytes, total_weight_bytes, resident_tensor_count
partitioned, pinned
```

`pinned` is a deployment-lifecycle guarantee, not CUDA page-locked host memory.
A ready partition is resident but not active. Activation pins it for admission;
explicit unload drains active work, destroys the engine/model objects, and
returns `model_memory` to `null`.

## Scope and limitations

- Initially supported architectures: `llama` and `qwen2`.
- Initially supported activation dtype: F32.
- The patch is tied to the pinned llama.cpp commit and must be revalidated when
  that pin changes.
- The residency byte counters cover model tensor payloads. mmap spans, allocator
  overhead, compute buffers, and context/KV allocations must be measured
  separately in physical-device telemetry.
- The colocated integration report records each runtime's Linux RSS, PSS, and
  GGUF mapping RSS under `process_memory`; these measurements remain separate
  from `model_memory` and must not be substituted for tensor-byte accounting.
- The default five-minute idle TTL is currently compiled into the stage adapter;
  exposing it as runtime configuration is a later operational improvement.
- Physical two-Jetson CUDA validation requires the target hardware and is not
  claimed by CPU CI.
