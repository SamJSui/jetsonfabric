# Runtime Stage Interface

The runtime stage interface is the engine-neutral boundary between JetsonFabric's binary stage protocol and an inference engine. Engine adapters do not handle HTTP or Stagewire objects directly.

## Boundary

```text
Stagewire frame
  -> protocol::StageRequest
  -> RuntimeService
  -> ModelManager
  -> StageWorker
  -> inference::StageInput
  -> LayerExecutor
  -> inference::StageOutput
  -> protocol::StageResponse
  -> Stagewire frame
```

`runtime/protocol` owns serialized metadata. `runtime/inference` owns typed execution semantics. `StageWorker` adapts between them. `ModelManager` owns the active model execution components used by `RuntimeService`.

## Identity and lifecycle

One generation has three identity levels:

```text
request ID       one user-facing completion
session ID       persistent generation state shared by all stages
stage request ID one operation on one stage
```

The coordinator creates the session ID. Prefill, decode, and cleanup retain that session ID while each stage operation receives its own request ID. A session remains bound to one model and stage plan until cleanup.

Each llama.cpp stage stores a context keyed by session ID. Explicit cleanup releases it; an idle timeout protects runtime memory when cleanup cannot complete.

## Typed payload transitions

```text
prefill first:        text/tokens   -> activation
prefill intermediate: activation    -> activation
prefill final:        activation    -> sampled_token

decode first:         sampled_token -> activation
decode intermediate:  activation    -> activation
decode final:         activation    -> sampled_token
```

For a one-stage pipeline, the same stage is first and final and produces a sampled token directly. Activations are little-endian row-major F32 tensors. Sampled tokens contain one 32-bit token ID. Logits and KV state remain engine-local.

## Current executors

The synthetic executor validates the same transitions without loading a model.

The llama.cpp stage executor:

- tokenizes only on the first stage;
- executes the configured layer range;
- exports hidden activations from non-final stages;
- imports activations on downstream stages;
- keeps final normalization, logits, and greedy sampling on the last stage;
- preserves stage-local contexts across decode steps;
- handles a one-stage full-range pipeline through the same stage interface.

## Current limitations

- The runtime starts with one configured active model; idle deployment lifecycle is not implemented.
- Chat completions are non-streaming and greedy-only.
- Partial-layer support is limited to Llama and Qwen2-family graphs.
- Inter-stage activations are F32.
- Each runtime still loads the complete GGUF.
- Runtime revision attestation and physical CUDA proof remain incomplete.
