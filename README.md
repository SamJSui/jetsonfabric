# JetsonMesh

JetsonMesh is an exo-inspired edge AI fabric for low-cost Jetson-class devices.

The goal is to turn multiple Jetson Orin Nano nodes into one observable serving
cluster with node discovery, model placement, benchmark-driven routing,
failover, and an OpenAI-compatible API surface.

This is not a local chatbot project. The core question is:

> Can a low-cost Jetson edge cluster maintain useful quality while improving
> throughput, latency, cost, reliability, or deployment flexibility versus a
> single node or cloud-only baseline?

## Initial Architecture

- Control plane: API gateway, node registry, model registry, scheduler, dashboard API.
- Jetson agent: reports node health, temperature, resources, queue state, and runtime capabilities.
- Runtime workers: future adapters for llama.cpp, TensorRT/ONNX, Triton, and Jetson AI Lab containers.
- Placement planner: decides whether to run single-node, replicated, layer-split, or cloud fallback.
- Benchmark service: records tokens/sec, p50/p95 latency, memory, power/thermal, quality, and failures.

## Quick Start

Start the control plane:

```powershell
py -m jetsonmesh_control.server --host 127.0.0.1 --port 52415 --join-token dev-token
```

Start a local test agent:

```powershell
py -m jetsonmesh_agent.agent --control-url http://127.0.0.1:52415 --join-token dev-token --node-id dev-node
```

Inspect cluster state:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:52415/healthz
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:52415/v1/nodes
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:52415/v1/routes/preview?model=qwen2.5-coder-1.5b-q4"
```

## Expanding To More Jetsons

Yes: eventually a new Jetson should be added by installing the agent and giving it
the control-plane URL plus a join token.

Expected join flow:

```bash
jetsonmesh-agent \
  --control-url http://beelink:52415 \
  --join-token <token> \
  --node-id jetson-02
```

The control plane should then discover its hardware profile, runtime capabilities,
and benchmark history before placing model work on it.

## Non-Goals For V0

- Do not claim a mini cluster beats frontier models.
- Do not start with tensor parallelism as the primary performance story.
- Do not make replicated serving the only feature.
- Do not require Kubernetes before the scheduler and benchmark loop work.

## First Credible Demo

1. One control plane running on the Beelink or local machine.
2. One Jetson agent reporting health and capabilities.
3. One model profile in the registry.
4. One benchmark result recorded for a local model backend.
5. Route preview explaining why a model should or should not run on a node.
6. A second Jetson added later to prove scaling, failover, and layer-split experiments.

