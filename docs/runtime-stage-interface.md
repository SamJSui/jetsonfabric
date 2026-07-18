# Runtime Stage Interface

The runtime stage interface is the engine-neutral boundary between JetsonFabric's
wire protocol and an inference-engine implementation. Engine adapters do not
receive or return HTTP or Stagewire objects.

## Boundary

```text
stagewire frame
  -> protocol::StageRequest
  -> pipeline_parallel::StageWorker
  -> inference::StageInput
  -> pipeline_parallel::LayerExecutor::execute
  -> inference::StageOutput
  -> protocol::StageResponse
  -> stagewire frame
```

`runtime/protocol` owns serialized metadata. `runtime/inference` owns typed
execution semantics. `StageWorker` adapts between the two.

## Identity and lifecycle

One generation has three levels of identity:

```text
chat/request ID
  -> one user-facing completion

session ID
  -> one persistent generation state shared by all stages

stage operation request ID
  -> one prefill/decode call to one stage
```

The coordinator generates a server-owned session ID. Prefill and every decode
step retain that session ID while receiving distinct request IDs for tracing.
The session remains bound to one immutable model and stage plan until cleanup.

Each llama.cpp stage stores a context keyed by session ID. Successful operations
refresh its last-used time. Explicit cleanup removes it immediately; a local
five-minute idle TTL reaps it if the coordinator or network disappears before
cleanup completes.

## Typed payload transitions

Payload-transition validation is mandatory for pipeline execution; there is no
public flag that disables it.

```text
prefill first:        text/tokens -> activation
prefill intermediate: activation  -> activation
prefill final:        activation  -> sampled_token

decode first:         sampled_token -> activation
decode intermediate:  activation    -> activation
decode final:         activation    -> sampled_token
```

Activations are raw little-endian row-major F32 tensors. Sampled tokens are one
little-endian 32-bit token ID. Logits and KV state remain inside the engine.

## Current executors

The synthetic executor validates the same semantic transitions without a real
model.

The active llama.cpp pipeline executor:

- tokenizes only on the first stage;
- executes the configured `[layer_start, layer_end)` range;
- exports hidden activations from non-final stages;
- imports activations on downstream stages;
- keeps logits and greedy sampling on the final stage;
- preserves stage-local llama.cpp contexts across decode steps.

A full-model llama.cpp executor remains only for single-stage compatibility. It
is not used by a multi-stage pipeline route.

## Public API path

`POST /v1/chat/completions` is coordinator-owned. Any JetsonFabric node forwards
that public request to the elected coordinator, which plans the ordered stages,
drives prefill and decode, and returns an OpenAI-style response.

`POST /v1/layer-split/run` remains a diagnostic endpoint. Both endpoints use the
same mandatory typed generation path.

## Current limitations

- Non-streaming chat completions only.
- Greedy sampling only.
- Llama and Qwen2-family partial-layer graphs only.
- F32 inter-stage activations only.
- Each runtime still loads the complete GGUF even though it executes one range.
- Exact runtime/model artifact identity and physical CUDA attestation remain to
  be added before the two-Jetson acceptance run.
