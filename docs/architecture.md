# Architecture

This page describes the current implementation. Current and target diagrams,
plus physical-hardware acceptance work, are documented in
[Architecture diagrams](architecture-diagrams.md) and the repository roadmap.

## Product shape

Every physical host runs one Go node supervising one node-local C++ runtime:

```text
jetsonfabric-node
  -> identity, discovery, membership, and coordinator election
  -> topology and stage planning
  -> OpenAI-compatible API and follower forwarding
  -> authenticated generation routing to stage 0
  -> node-local jetsonfabric-runtime-worker
      -> ModelManager
      -> GenerationRunner
      -> HTTPStageClient
      -> StageWorker
      -> inference-engine LayerExecutor
      -> llama.cpp or synthetic adapter
```

The runtime worker is JetsonFabric infrastructure. `llama.cpp` is the first real inference engine, and CPU or CUDA is the compute backend.

## Control plane

The Go node owns cluster policy:

- `internal/discovery` finds static or mDNS peers;
- `internal/membership` maintains a fresh local member view;
- `internal/election` deterministically selects a coordinator;
- `internal/facade` lets clients call any node and forwards coordinator-owned APIs;
- `internal/clusterplan` selects compatible runtimes, orders stages, and assigns contiguous layer ranges;
- `internal/coordinator` owns public APIs, deployment admission, route planning,
  and response translation.

Membership is a local peer cache, not a consensus database. Planning uses a fresh snapshot for each request.

## Current generation data plane

The coordinator makes one authenticated generation request to the runtime on
stage 0. That runtime is the deterministic pipeline-session leader:

```text
client
  -> any node
  -> elected coordinator
  -> stage-0 node /v1/runtime/generate
  -> node-local runtime /v1/generate
  -> GenerationRunner
      -> local ModelManager for stage 0
      -> HTTPStageClient -> authenticated peer node facade -> peer runtime
  -> NDJSON token and done events
  -> buffered OpenAI response or incremental SSE
```

For prefill, the first stage accepts text or tokens. Non-final stages emit F32 hidden activations. The final stage keeps logits local and emits one sampled token. Decode repeats the same ordered stage path using persistent stage-local contexts.

`internal/stagewire` and `runtime/protocol` implement the matching versioned
binary frame. `internal/runtimebridge` proxies generation and stage traffic
between each Go node and its local runtime. The C++ `GenerationRunner` validates
stage responses and forwards each output to the next stage without returning
the payload to the coordinator. `internal/stageexec` remains available for the
diagnostic `/v1/layer-split/run` API.

## Runtime ownership

```text
RuntimeService
  -> ModelManager
      -> resident deployments keyed by immutable deployment ID and epoch
          -> ready replacement epoch
          -> active admission epoch
          -> draining old epoch for pinned sessions
          -> model identity
          -> stage-local model tensor residency
          -> InferenceEngineParts
          -> StageWorker
              -> LayerExecutor
  -> GenerationRunner
      -> local ModelManager stage invocation
      -> HTTPStageClient for peer stages
```

`ModelManager` is the ownership boundary for model execution state. A runtime can
start idle and temporarily retain more than one epoch so a replacement partition
can reach `ready` without evicting the active partition. The coordinator
activates every replacement stage, atomically changes admission to that epoch,
marks the previous epoch `draining`, and unloads it only after its admission
count reaches zero.

## Stage and session contract

Stage position is count-based:

```text
0 <= stage_index < stage_count
is_first = stage_index == 0
is_last  = stage_index == stage_count - 1
```

For `stage_count=1`, stage `0` is both first and final. Layer ranges must be contiguous and exhaustive for the selected pipeline.

One completion has:

```text
request ID  -> user-facing completion
session ID  -> persistent generation state across all stages
stage request ID -> one prefill, decode, or cleanup operation on one stage
```

The coordinator creates the session ID and pins deployment admission for the
request. The stage-0 runtime closes every stage session on success, failure, or
downstream cancellation. Runtime adapters also reap sessions after an idle
timeout.

Managed requests carry the deployment ID, epoch, model ID, and artifact SHA-256
through every stage request and response. Each runtime rejects a stage operation
whose identity does not select an active or draining deployment. Peer node
facades also require `JETSONFABRIC_CLUSTER_TOKEN`; local runtime proxies strip
that credential before forwarding to the loopback runtime.

## Compatibility and topology

A pipeline is grouped by correctness-critical runtime identity, including engine adapter, model ID, exact model artifact hash, and execution mode. Compute backend remains placement and telemetry metadata so compatible CPU and CUDA stages can share the same semantic contract.

Topology describes physical placement:

- `colocated`: multiple logical stages share one physical host;
- `distributed`: selected stages run on distinct physical hosts.

## Current limitations

- Stage-local tensor loading currently supports only Llama and Qwen2 at the
  pinned `llama.cpp` revision.
- Residency accounting covers tensor payload bytes, not allocator, compute, or
  context/KV overhead.
- Safe rebalance needs temporary memory for old and new partitions on reused
  nodes; insufficient overlap capacity causes rollback to the old epoch.
- Reconciliation state is not replicated across coordinator failover yet.
- Inter-stage activations are F32.
- Runtime peer calls currently use sequential HTTP/1.1 connections through node
  facades; connection reuse and overlapped microbatches are not implemented.
- Peer authentication uses one shared bearer token over plaintext HTTP. TLS,
  per-node credentials, and secure admission are not implemented.
- Cancellation is checked between stage passes and stream writes; it cannot
  interrupt a blocking peer request before that request's transport timeout.
- Chat completions support buffered and SSE responses but are greedy-only.
- Physical multi-Jetson CUDA acceptance has not yet been completed.
