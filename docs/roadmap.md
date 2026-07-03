# Roadmap

JetsonFabric now tracks four major milestones:

```text
POC: one full-model replica deployed onto one node and serving
P0/MVP: real layer-split inference across Jetson nodes
P1: tensor-parallelism experiment
P2: operational edge fabric and measured runtime optimization
```

The naming matters. A single Jetson serving a full model is a **single full-model
replica**, not distributed inference. It is still necessary because it proves the
node, runtime, model artifact, routing, and benchmark loop before the project
attempts multi-node execution.

## Foundation - Scaffold

- Go control plane API
- Go node heartbeat endpoint
- Go agent heartbeat
- Model registry
- Route preview endpoint
- Agent proxy to an OpenAI-compatible backend
- Synthetic layer-split planning and transport smoke tests

## POC - Single Full-Model Replica Serving

Goal: deploy one model onto one Jetson-class node and serve it through
JetsonFabric.

- Run one Jetson agent.
- Detect Jetson hardware, JetPack/CUDA/TensorRT hints, memory, temperature, and
  runtime capabilities.
- Ensure one model artifact exists on the node.
- Start a node-local runtime such as `llama-server`.
- Have the agent advertise exactly one ready model/backend.
- Route one prompt through `jetsonfabric-control` to that backend.
- Return a model response through `/v1/chat/completions`.
- Record tokens/sec, p50/p95 latency, time-to-first-token, memory, and
  temperature.

The POC is complete when a fresh setup can build the agent/control binaries,
install the agent on one Jetson, send one prompt through the control plane, and
show a benchmark record for that run.

## P0/MVP - Layer-Split Inference

Goal: make the first real distributed inference path work.

- Use two Jetsons first; add a third only after the two-node path is measured.
- Build or integrate a shard-capable runtime. Stock `llama-server` is not enough
  because it cannot expose intermediate layer activations.
- Split a small transformer by layer ranges across Jetsons.
- Each runtime worker owns its assigned layers, local runtime state, and KV-cache
  behavior for its stage.
- Send activation tensors and metadata across the network during prefill and
  decode.
- Keep replica_serving as a baseline/control, not the main novelty.
- Start with ordinary TCP over the built-in network before buying faster
  transport hardware.
- Compare POC single-replica serving, replica_serving, and pipeline_parallel routes on
  the same prompt sets.

P0/MVP is complete when JetsonFabric returns a real model response from a
multi-node layer-split execution path and records route metadata with nodes,
layer ranges, transport, bytes moved, latency, memory, and thermal data.

## P1 - Tensor Parallelism

Goal: test whether splitting tensor operations across Jetson nodes can help
enough to justify its synchronization cost.

- Prototype a narrow tensor-parallel path for one model component.
- Measure synchronization frequency, network bytes/token, copy overhead, latency,
  and GPU utilization.
- Compare against the POC single-replica baseline and the P0/MVP layer-split
  path.
- Treat a negative result as valid if the measurements show communication cost
  dominates.

P1 is complete when JetsonFabric can explain, with measurements, whether tensor
parallelism is useful on the target Jetson network for the selected model class.

## P2 - Operational Edge Fabric

Goal: turn the runtime experiments into a repeatable, observable edge serving
system.

- Agent-managed model artifact download, verification, and runtime launch.
- Persistent control-plane state for nodes, deployments, benchmarks, and model
  metadata.
- Authenticated admin APIs for model and deployment management.
- Dashboard/API views for node health, route decisions, runtime readiness, and
  benchmark history.
- Profile-driven placement using memory, thermal, latency, and network data.
- Failover and degraded routing when a node or runtime disappears.
- Runtime transport optimization only where benchmarks identify a bottleneck:
  activation compression, pipelining, 10GbE TCP, pinned buffers, and eventually
  RDMA if justified.

P2 is complete when JetsonFabric can deploy, observe, route, recover, and explain
model work across the cluster without hand-tuning each run.

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
