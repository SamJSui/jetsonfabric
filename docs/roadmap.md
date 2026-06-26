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

## P1 - Single Jetson Observability

- Add dashboard shell.
- Show node health, model compatibility, route preview, and latest benchmark.
- Explain why the single-Jetson route is selected or rejected.
- Add authenticated join-token enforcement.

## P2 - Two Jetson Baseline

- Add second Jetson.
- Implement replica baseline mode.
- Measure throughput scaling and failover.
- Keep this as benchmark control, not the main novelty.

## P3 - Layer-Split Runtime

- Split a small transformer by layers across two Jetsons.
- Each worker stores its assigned layers and KV cache.
- Send hidden states across the network during prefill/decode.
- Measure latency, network bytes/token, memory, and temperature.
- Prototype can be simple first; optimized runtime pieces should move into C++.

## P4 - Profile-Driven Placement

- Profile per-layer latency on each node.
- Move split point based on node speed, thermals, and memory.
- Compare local, replica, layer-split, and cloud fallback.

## P5 - Transfer Optimization

- Compare FP16, BF16, and INT8 activation transfer.
- Add optional compression.
- Use pinned buffers where possible.
- Quantify whether transfer optimization helps enough to matter.

## P6 - OpenAI-Compatible Gateway

- Implement `/v1/chat/completions`.
- Stream responses from the selected route.
- Return route metadata for observability.

## Python Boundary

Python is allowed for benchmark analysis, graph generation, and reports. It
should not own the production control plane, node agent, or distributed runtime.
