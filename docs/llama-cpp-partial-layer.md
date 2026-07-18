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
2. split prefill through two stage adapters;
3. one split decode step using persistent stage contexts;
4. explicit session cleanup;
5. idle-session TTL cleanup.

The first and second sampled tokens must match the full-model baseline.

The two-node CPU E2E additionally requires the real activation to cross the
existing Stagewire path byte-for-byte before the final runtime samples. The
coordinator owns the multi-token decode loop and gives each stage operation a
unique request ID while retaining one server-generated session ID.

## Scope and limitations

- Initially supported architectures: `llama` and `qwen2`.
- Initially supported activation dtype: F32.
- The patch is tied to the pinned llama.cpp commit and must be revalidated when
  that pin changes.
- The current llama.cpp context may still reserve memory structures for the full
  model even though graph execution and memory updates are restricted to the
  assigned range. Memory-footprint reduction must be measured separately.
- The default five-minute idle TTL is currently compiled into the stage adapter;
  exposing it as runtime configuration is a later operational improvement.
- Physical two-Jetson CUDA validation requires the target hardware and is not
  claimed by CPU CI.
