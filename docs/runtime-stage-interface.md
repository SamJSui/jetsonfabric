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

For a complete chat request, `RuntimeService::generate` creates a
`GenerationRunner`. It invokes stage 0 through the local `ModelManager` and
later stages through `HTTPStageClient`, using this same stage boundary for every
prefill, decode, and cleanup operation.

## Identity and lifecycle

One generation has three identity levels:

```text
request ID       one user-facing completion
session ID       persistent generation state shared by all stages
stage request ID one operation on one stage
```

The coordinator creates the session ID and sends it once to the stage-0 runtime.
The runtime generation leader retains that ID through prefill, decode, and
cleanup while each stage operation receives its own request ID. A session
remains bound to one model and immutable stage plan until cleanup.

For coordinator-managed deployments, every operation also carries the exact
deployment ID, epoch, model ID, and model SHA-256. `ModelManager` validates that
identity before execution or cleanup, and the generation leader rejects a
response that does not echo it unchanged.

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

- The runtime supports idle startup and explicit load, activate, status, and
  unload operations, but automatic reconciliation is not implemented.
- Chat completions support buffered and SSE output but are greedy-only.
- Cancellation is observed between stage passes or failed stream writes; an
  already-blocking peer request runs until its transport timeout.
- Peer stage requests require the shared cluster token but do not use TLS.
- Partial-layer support is limited to Llama and Qwen2-family graphs.
- Inter-stage activations are F32.
- Llama and Qwen2 runtimes load only the model tensors assigned to their stage;
  context/KV and allocator overhead are accounted separately.
- Physical multi-Jetson CUDA proof remains incomplete.
