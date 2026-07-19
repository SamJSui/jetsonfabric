# Architecture

This page describes the current implementation. Future deployment, generation, and rebalance designs are documented separately in [Architecture diagrams](architecture-diagrams.md) and the repository roadmap.

## Product shape

Every physical host runs one Go node supervising one node-local C++ runtime:

```text
jetsonfabric-node
  -> identity, discovery, membership, and coordinator election
  -> topology and stage planning
  -> OpenAI-compatible API and follower forwarding
  -> ordered stage execution
  -> node-local jetsonfabric-runtime-worker
      -> ModelManager
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
- `internal/coordinator` owns public APIs and request-level generation orchestration.

Membership is a local peer cache, not a consensus database. Planning uses a fresh snapshot for each request.

## Current generation data plane

The coordinator currently owns both the token loop and stage loop:

```text
client
  -> any node
  -> elected coordinator
  -> stageexec
  -> target node API
  -> target runtimebridge
  -> target runtime StageWorker
  -> assigned LayerExecutor
  -> Stagewire response
  -> next stage
```

For prefill, the first stage accepts text or tokens. Non-final stages emit F32 hidden activations. The final stage keeps logits local and emits one sampled token. Decode repeats the same ordered stage path using persistent stage-local contexts.

`internal/stagewire` and `runtime/protocol` implement the matching versioned binary frame. `internal/runtimebridge` proxies between a Go node and its local runtime. `internal/stageexec` validates and forwards each stage output as the next stage input.

Direct runtime-to-runtime transport and a single runtime `Generate` call are target architecture, not current behavior.

## Runtime ownership

```text
RuntimeService
  -> ModelManager
      -> active LoadedModel
          -> model identity
          -> InferenceEngineParts
          -> StageWorker
              -> LayerExecutor
```

`ModelManager` is now the ownership boundary for active model execution state. The runtime still starts with exactly one configured model and assignment; idle startup and prepare/activate/unload operations are not implemented yet.

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

The coordinator creates the session ID and closes every stage session when generation ends. Runtime adapters also reap sessions after an idle timeout.

## Compatibility and topology

A pipeline is grouped by correctness-critical runtime identity, including engine adapter, model ID, exact model artifact hash, and execution mode. Compute backend remains placement and telemetry metadata so compatible CPU and CUDA stages can share the same semantic contract.

Topology describes physical placement:

- `colocated`: multiple logical stages share one physical host;
- `distributed`: selected stages run on distinct physical hosts.

## Current limitations

- Every runtime still opens the complete GGUF even though it executes only its assigned transformer range.
- Runtime deployment is fixed at process startup.
- Go coordinates every stage operation and autoregressive decode step.
- Inter-stage activations are F32.
- Chat completions are non-streaming and greedy-only.
- Physical multi-Jetson CUDA acceptance has not yet been completed.
