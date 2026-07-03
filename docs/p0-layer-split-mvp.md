# P0/MVP - Layer-Split Inference

P0/MVP is the first real distributed inference milestone. The goal is to split a
single model by layer ranges across two Jetson Orin-class nodes and return a
valid response through the JetsonFabric API.

This is the point where JetsonFabric becomes more than control-plane proxying.
The POC proves one node can serve a full model. P0/MVP proves multiple nodes can
cooperate on one inference request.

## Goal

Run a small transformer across two Jetsons:

```text
dopey:  embeddings + layers 0..N
grumpy: layers N+1..final + lm_head
```

The first implementation can be narrow and explicit. It should prioritize
correctness, observability, and measurement over generalized runtime flexibility.

## Required Capabilities

- Control plane loads model metadata, including layer count.
- Agents report runtime readiness and usable hardware facts.
- Control plans a `pipeline_parallel` route with concrete layer ranges.
- A shard-capable runtime process runs on each participating node.
- Runtime A sends activation tensors and metadata to Runtime B.
- The final response returns through `/v1/chat/completions`.
- Benchmarks record latency, tokens/sec, memory, temperature, and network bytes.
- The same prompt set can compare POC single-node serving against layer split.

## Runtime Boundary

Stock `llama-server` is not enough for this milestone because it serves a whole
model behind an OpenAI-compatible API. P0/MVP needs a runtime that can:

- load or address a model shard;
- execute only assigned layer ranges;
- emit and receive activation tensors;
- own KV-cache behavior for its stage;
- expose a small runtime control/health API.

The runtime path should be C++ first. Go still owns orchestration, APIs,
heartbeats, placement, and benchmark records.

## Acceptance Criteria

P0/MVP is complete when JetsonFabric can show:

1. Two Jetson agents registered and healthy.
2. A model deployment split across those nodes.
3. One prompt routed through a real layer-split runtime path.
4. A response that comes from the split execution path, not a synthetic stage.
5. Route metadata listing both nodes, layer ranges, transport, and bytes moved.
6. Benchmark output comparing single-node POC serving to layer split.

## Non-Goals

- Tensor parallelism.
- RDMA or GPUDirect.
- General multi-model scheduling.
- Claims that layer split is faster before benchmark evidence exists.
