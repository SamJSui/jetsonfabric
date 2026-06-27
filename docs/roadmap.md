# Roadmap

## Foundation - Scaffold

- Go control plane API
- Go node heartbeat endpoint
- Go agent heartbeat
- Model registry
- Route preview endpoint

## P0 - Single Jetson Model Serving

Current priority: prove that JetsonFabric can run a real model on one Jetson
through the control-plane path. Layer split waits until this works.

- Run one Jetson agent.
- Detect Jetson hardware, JetPack/CUDA/TensorRT hints, memory, temperature, and
  runtime capabilities.
- Deploy one small LLM through a local Jetson backend.
- Route one prompt through `jetsonfabric-control` to that backend.
- Return a model response through `/v1/chat/completions`.
- Record tokens/sec, p50/p95 latency, time-to-first-token, memory, and
  temperature.
- Store benchmark results in a simple local format.
- Produce a repeatable demo script.

P0 is complete when a fresh checkout can build the agent/control binaries,
install the agent on one Jetson, send one prompt through the control plane, and
show a benchmark record for that run.

## P1 - Multi-Jetson Layer-Split Inference

Goal: use two Jetsons first, then three, to test whether layer-split model
parallelism can run a larger or better model than one Jetson can comfortably
serve.

- Add a second Jetson, then a third only after the two-node path is measured.
- Keep replica_serving as a baseline/control, not the main novelty.
- Split a small transformer by layer ranges across Jetsons.
- Each worker owns its assigned layers, local runtime state, and local KV-cache
  responsibility for its shard.
- Send hidden states across the network during prefill and decode.
- Start with ordinary TCP over the built-in network before buying faster
  transport hardware.
- Measure latency, time-to-first-token, tokens/sec, network bytes/token, memory,
  temperature, and error behavior.
- Compare single-node, replica_serving, and layer_split routes on the same
  prompt sets.

P1 is complete when JetsonFabric can show whether layer split improves at least
one serious metric:

- a larger model fits;
- coding or reasoning pass rate improves enough to justify added latency;
- pipeline throughput improves under concurrent load;
- memory pressure per node drops enough to matter.

## P2 - Distributed Runtime Optimization

Goal: make distributed inference more optimal after P1 proves the layer-split
path and identifies the bottleneck.

P2 is primarily C++/CUDA runtime optimization work, not Go control-plane work.
P1 may introduce a minimal C++ layer worker if the selected model runtime
requires it. Go still owns orchestration, scheduling, APIs, and benchmark
records. C++ owns latency-sensitive runtime workers, tensor framing, transport
backends, and integration with CUDA/TensorRT/llama.cpp.

Planned optimization ladder:

- Profile the P1 layer-split path and identify compute, copy, serialization, or
  network bottlenecks.
- Implement a stable activation/tensor wire protocol with explicit shape, dtype,
  byte length, request/session ID, layer boundary, and decode step fields.
- Keep dtype as a typed enum in code with stable numeric wire values.
- Compare FP16, BF16, INT8, and packed INT4 activation transfer where quality
  impact can be measured.
- Use pinned or mapped buffers where they reduce CPU/GPU copy overhead.
- Compare built-in 1GbE TCP against optional 10GbE TCP.
- Evaluate RDMA only after TCP and 10GbE measurements show transport overhead is
  the limiting factor.
- Treat tensor parallelism as an experiment after transport numbers justify it,
  because it requires much more frequent synchronization than layer split.

P2 is complete when JetsonFabric can explain the bottleneck with measurements
and show whether transport/runtime changes materially improve latency,
throughput, memory fit, energy, or quality-per-latency.

## P3 - Profile-Driven Placement

- Profile per-layer latency on each node.
- Move split point based on node speed, thermals, and memory.
- Compare local, replica_serving, layer_split, and cloud fallback.

## Supporting Work - Observability And Gateway

- Keep `/v1/chat/completions` compatible with OpenAI-style clients.
- Stream responses from the selected route.
- Return route metadata for observability.
- Add dashboard views for node health, model compatibility, route preview,
  latest benchmarks, thermal behavior, and route rejection reasons.
- Add authenticated join-token enforcement.

## Python Boundary

Python is allowed for benchmark analysis, graph generation, and reports. It
should not own the production control plane, node agent, or distributed runtime.
