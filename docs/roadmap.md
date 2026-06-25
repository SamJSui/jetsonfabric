# Roadmap

## Phase 0 - Scaffold

- Control plane API
- Node heartbeat endpoint
- Agent heartbeat
- Model registry
- Route preview endpoint

## Phase 1 - Single Jetson Baseline

- Run one Jetson agent.
- Deploy one small LLM through llama.cpp or Jetson AI Lab container.
- Record tokens/sec, p50/p95 latency, memory, and temperature.
- Add dashboard shell.

## Phase 2 - Two Jetson Baseline

- Add second Jetson.
- Implement replica baseline mode.
- Measure throughput scaling and failover.
- Keep this as benchmark control, not the main novelty.

## Phase 3 - Layer-Split Runtime

- Split a small transformer by layers across two Jetsons.
- Each worker stores its assigned layers and KV cache.
- Send hidden states across the network during prefill/decode.
- Measure latency, network bytes/token, memory, and temperature.

## Phase 4 - Profile-Driven Placement

- Profile per-layer latency on each node.
- Move split point based on node speed, thermals, and memory.
- Compare local, replica, layer-split, and cloud fallback.

## Phase 5 - Transfer Optimization

- Compare FP16, BF16, and INT8 activation transfer.
- Add optional compression.
- Use pinned buffers where possible.
- Quantify whether transfer optimization helps enough to matter.

## Phase 6 - OpenAI-Compatible Gateway

- Implement `/v1/chat/completions`.
- Stream responses from the selected route.
- Return route metadata for observability.

