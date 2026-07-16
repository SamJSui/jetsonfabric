# Architecture

See [Architecture Diagrams](architecture-diagrams.md) for visual views of these contracts.

## Product shape

Every logical node runs one Go process:

```text
jetsonfabric-node
  -> identity, membership, discovery, and election
  -> cluster planning and coordinator APIs
  -> runtime bridge
  -> node-local jetsonfabric-runtime-worker
      -> inference-engine adapter such as llama.cpp
```

The runtime worker is a JetsonFabric-owned host process. The engine is the model implementation it hosts, and the compute backend is CPU or CUDA.

## Main boundaries

1. `cmd/jetsonfabric-node` parses flags and starts the node app.
2. `internal/node` owns process lifecycle and runtime supervision.
3. `internal/membership` and `internal/discovery` maintain the local peer view.
4. `internal/election` selects a coordinator; `internal/facade` provides the any-node API.
5. `internal/clusterplan` selects eligible nodes, orders stages, assigns layer ranges, and reports topology.
6. `internal/inference` defines prefill/decode lifecycle and legal payload semantics.
7. `internal/stagewire` defines and validates the versioned binary stage frame.
8. `internal/stageexec` executes an ordered pass and carries each binary payload into the next stage.
9. `internal/runtimebridge` streams node-local stage frames to the supervised runtime.
10. `runtime/` hosts C++ engine adapters and implements the matching frame contract.

The package names answer separate questions:

```text
inference:       what should a distributed inference session mean?
clusterplan:     where should each stage run?
stagewire:       what crosses the stage boundary?
stageexec:       how is an ordered stage pass invoked?
runtimebridge:   how does the Go node reach its local C++ runtime?
```

## Execution vocabulary

Execution mode describes how inference work is distributed:

- `data_parallel`: each replica owns a complete model.
- `pipeline_parallel`: ordered transformer layer ranges are assigned to stages.
- `tensor_parallel`: tensor operations are partitioned across devices or nodes.

Stage position is represented only by `stage_index` and `stage_count`:

```text
is_first = stage_index == 0
is_last  = stage_index == stage_count - 1
```

For `stage_count=1`, index `0` is both first and last. Layer ranges are contiguous and exhaustive.

Topology describes physical placement:

- `colocated`: stages share a physical host.
- `distributed`: stages span physical hosts.

The stage wire contract supports `text`, `tokens`, `activation`, and `sampled_token`. Logits and KV cache remain engine-local.

## Node lifecycle

A node binds its API listener, derives its advertised URL, starts or attaches to its runtime worker, constructs the facade/coordinator/runtime-bridge stack, publishes itself to membership, starts discovery, and serves until shutdown.

`membership.Store` is a local peer cache, not a consensus database. Planning takes a fresh snapshot for each request.

## Request routing

Clients can call any node. Coordinator-owned requests are served locally by the elected coordinator or proxied to its `APIURL`. `POST /v1/layer-split/stage` remains node-local and is forwarded through `runtimebridge` to that node's runtime URL.

## Stage data plane

One stage operation uses `stagewire` v1:

```text
fixed binary header
  -> versioned JSON metadata
  -> raw payload bytes
```

The frame uses media type `application/vnd.jetsonfabric.stage.v1+octet-stream`. Tensor bytes are not base64-encoded. Metadata carries dtype, shape, little-endian byte order, row-major layout, byte length, and CRC32. See [Stagewire v1](stagewire-v1.md).

The coordinator-mediated P0 path is:

```text
runtime stage 0
  -> node 0 runtimebridge
  -> stageexec at coordinator
  -> node 1 API
  -> node 1 runtimebridge
  -> runtime stage 1
```

Direct runtime-to-runtime transport may replace this path later without changing the stagewire payload semantics.

## Current execution status

The existing llama.cpp path still performs full-model generation at each stage. Its inter-stage semantic payload remains `text`, although transport now uses the binary stagewire frame.

A synthetic CI engine proves the activation data plane independently of llama.cpp:

```text
text
  -> synthetic stage 0
  -> f32[4,16] activation bytes
  -> second logical node and runtime
  -> u32 sampled-token payload containing activation CRC32
```

CI asserts byte length and checksum continuity across the logical-node boundary. This proves activation transport, not partial transformer execution.

## P0 partial-layer target

```text
text
  -> tokenizer
  -> first assigned layer range
  -> activation stagewire payload
  -> remaining assigned layer ranges
  -> final-stage logits and sampling
  -> sampled token
```

Each stage keeps KV state for its assigned layers local across decode steps. The remaining engine milestone is to make llama.cpp or another adapter produce and consume the real activation tensors defined by this data plane.
