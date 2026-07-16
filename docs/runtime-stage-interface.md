# Runtime Stage Interface

The runtime stage interface is the engine-neutral boundary between JetsonFabric's
wire protocol and an inference-engine implementation. Engine adapters no longer
receive or return `protocol::StageRequest` or `protocol::StageResponse` values.

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

`runtime/protocol` owns framing and serialized metadata. `runtime/inference` owns
typed execution semantics. `pipeline_parallel::StageWorker` is the adapter between
those two boundaries.

## Typed input

`inference::StageInput` contains:

- session, request, and model identity;
- `Phase::Prefill` or `Phase::Decode`;
- monotonic decode step;
- count-based stage position;
- assigned transformer layer range;
- a typed payload;
- generation limits such as `max_tokens`.

A payload uses `PayloadKind::Text`, `Tokens`, `Activation`, or `SampledToken`.
Text owns UTF-8 bytes. Tensor payloads own dtype, shape, byte order, layout, and
raw bytes.

The execution interface intentionally excludes HTTP status, media type, frame
version, CRC32, and node-routing information. Those are transport concerns.

## Typed output and errors

`inference::StageOutput` contains only the produced typed payload and engine token
counts. `inference::ExecutionResult` separates invalid engine input from an engine
execution failure. `StageWorker` maps those classes to the current HTTP-facing
status and constructs the wire response.

The worker measures latency and derives byte counts. Engines do not construct
wire responses or stage traces.

## Current executors

The synthetic executor implements the target semantic transitions:

```text
first stage:        text/tokens or sampled token -> activation
intermediate stage: activation -> activation
final stage:        activation -> sampled token
```

The llama.cpp full-model executor is an explicit compatibility adapter. It accepts
prefill text and returns text until a real partial-layer adapter can produce and
consume activation tensors. That compatibility behavior is isolated behind the
same typed interface rather than leaking wire structs into the engine adapter.

## Next engine milestone

A partial-layer engine adapter can now implement the same interface without
changing Stagewire, StageWorker, runtime bridging, or coordinator transport:

```text
StageInput(tokens or sampled_token)
  -> execute assigned layer range
  -> StageOutput(activation)

StageInput(activation)
  -> execute assigned layer range
  -> StageOutput(activation or sampled_token)
```
