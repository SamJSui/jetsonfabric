# JetsonFabric

JetsonFabric is an exo-inspired edge AI fabric for low-cost Jetson-class devices.

The goal is to turn multiple Jetson Orin Nano nodes into one observable serving
cluster with node discovery, model placement, benchmark-driven routing,
failover, and an OpenAI-compatible API surface.

This is not a local chatbot project. The core question is:

> Can a low-cost Jetson edge cluster maintain useful quality while improving
> throughput, latency, cost, reliability, or deployment flexibility versus a
> single node or cloud-only baseline?

## Initial Architecture

- Go control plane: API gateway, node registry, model registry, scheduler,
  dashboard API.
- Go Jetson agent: reports node health, temperature, resources, queue state, and
  runtime capabilities.
- C++ runtime lane: future adapters for llama.cpp, TensorRT/ONNX, Triton,
  layer-shard transport, and Jetson AI Lab containers.
- Placement planner: decides whether to run single-node, replicated,
  layer-split, or cloud fallback.
- Benchmark service: records tokens/sec, p50/p95 latency, memory,
  power/thermal, quality, and failures.
- Python: benchmark analysis, graphs, and reports only.

Distributed runtime phases:

- P0: one Jetson, one real model backend, one routed prompt, one benchmark.
- P1: two to three Jetsons with layer-split inference; replica serving is only
  the control baseline.
- P2: C++/CUDA runtime optimization, including activation transport,
  compression, 10GbE TCP, optional RDMA, and tensor-parallel experiments only
  after measurements justify them.

## Project Context

- [AGENTS.md](AGENTS.md) captures coding-agent instructions and project boundaries.
- [docs/project-context.md](docs/project-context.md) captures the pitch, value-add,
  hardware strategy, and honest performance story.
- [docs/p0-single-jetson.md](docs/p0-single-jetson.md) captures the current
  priority: get one real model serving on one Jetson before distributed runtime
  work.
- [docs/roadmap.md](docs/roadmap.md) captures the P0/P1/P2 progression from
  single-Jetson serving to layer split and later transport optimization.
- [docs/engineering-standards.md](docs/engineering-standards.md) captures the
  required implementation quality bar.
- [docs/desktop-multi-agent.md](docs/desktop-multi-agent.md) captures the local
  multi-agent simulation and benchmark workflow.

## Quick Start

Download the local desktop smoke assets:

```sh
sh scripts/download-p0-desktop-model.sh
```

Download any configured model artifact by ID:

```sh
sh scripts/download-model-artifact.sh --model-id qwen2.5-coder-3b-q4
```

Start the local llama.cpp backend:

```sh
sh scripts/run-local-llama.sh --background
```

Start the control plane from WSL/Linux:

```sh
sh scripts/run-control.sh --background
```

Start a local test agent:

```sh
sh scripts/run-agent.sh \
  --node-id dev-node \
  --advertise-url http://127.0.0.1:52416 \
  --llama-url http://127.0.0.1:8080 \
  --llama-models qwen2.5-coder-1.5b-q4 \
  --background
```

Inspect cluster state:

```sh
curl -sS http://127.0.0.1:52416/healthz
curl -sS http://127.0.0.1:52415/healthz
curl -sS http://127.0.0.1:52415/v1/nodes
curl -sS "http://127.0.0.1:52415/v1/routes/preview?model=qwen2.5-coder-1.5b-q4"
```

Send one prompt through JetsonFabric to the agent-proxied local model backend:

```sh
curl -sS -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/p0-local-smoke/chat-request.json
```

The local P0 request path is shown in
[docs/p0-local-smoke-sequence.svg](docs/p0-local-smoke-sequence.svg).

### Desktop Multi-Agent Simulation

Before the Jetson arrives, run multiple desktop agents against the same local
llama.cpp backend:

```sh
sh scripts/run-desktop-agents.sh --count 3
```

Inspect the generated layer-split plan:

```sh
curl -sS "http://127.0.0.1:52415/v1/layer-split/plan?model=qwen2.5-coder-1.5b-q4"
```

Run a synthetic layer-split prompt through all planned agents:

```sh
curl -sS -X POST http://127.0.0.1:52415/v1/layer-split/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/p0-local-smoke/chat-request.json
```

This tests discovery, heartbeat registration, agent proxying, route planning,
stage transport, and benchmark recording. It does not execute real distributed
transformer layers yet; all desktop simulation agents can point at the same
local model runtime.

Run a repeatable desktop chat benchmark through the control plane:

```sh
sh scripts/bench-desktop-chat.sh --count 5 --concurrency 1
```

The control plane appends request records to `data/benchmarks.jsonl`, and the
benchmark command writes a client-side summary to
`data/desktop-chat-benchmark.json`.

Stop the desktop simulation agents:

```sh
sh scripts/stop-desktop-agents.sh --count 3
```

### Agent Container

The agent can be built as a small container image:

```sh
docker build -f Dockerfile.agent -t jetsonfabric-agent:dev .
```

For Jetson targets, build an arm64 image with Docker Buildx:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.agent -t jetsonfabric-agent:arm64 .
```

## Expanding To More Jetsons

Yes: eventually a new Jetson should be added by installing the agent and giving it
the control-plane URL plus a join token.

Expected join flow:

```bash
./jetsonfabric-agent \
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
3. One real local model backend running on the Jetson.
4. One prompt routed through JetsonFabric to that model.
5. One benchmark result recorded for the local model backend.
6. Route preview explaining why the model should or should not run on the node.
7. A second Jetson added later to prove scaling, failover, and layer_split.

## Build

Runtime services are Go-native:

```sh
sh scripts/build.sh
```

This produces Linux development binaries and Linux arm64 binaries for Jetson.
See [docs/build.md](docs/build.md).
