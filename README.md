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

- POC: one Jetson, one full-model replica, one routed prompt, one benchmark.
- P0/MVP: two Jetsons with real layer-split inference.
- P1: tensor-parallelism experiment, measured against the POC and layer-split
  baselines.
- P2: operational edge fabric with model lifecycle, persistent state,
  profile-driven placement, failover, dashboard/API visibility, and measured
  runtime optimization.

## Project Context

- [AGENTS.md](AGENTS.md) captures coding-agent instructions and project boundaries.
- [docs/project-context.md](docs/project-context.md) captures the pitch, value-add,
  hardware strategy, and honest performance story.
- [docs/poc-single-node-serving.md](docs/poc-single-node-serving.md) captures the
  proof of concept: get one full-model replica serving on one Jetson.
- [docs/p0-layer-split-mvp.md](docs/p0-layer-split-mvp.md) captures the MVP:
  real layer-split inference across Jetson nodes.
- [docs/roadmap.md](docs/roadmap.md) captures the POC/P0/P1/P2 progression from
  single-node serving to layer split, tensor-parallel research, and operational
  fabric work.
- [docs/engineering-standards.md](docs/engineering-standards.md) captures the
  required implementation quality bar.

## Quick Start - Real Jetson POC

The current target node is `dopey`. The POC path is:

```text
control plane on dev machine or Beelink
  -> JetsonFabric agent on dopey
  -> agent proxy on dopey
  -> local OpenAI-compatible model backend on dopey
  -> benchmark JSONL on the control machine
```

### 1. Build

From the repo root on the development machine:

```sh
sh scripts/build.sh
```

This builds Linux development binaries and Linux arm64 binaries for Jetson.

### 2. Check dopey

Verify the Jetson is reachable and has the expected hardware/runtime basics:

```sh
sh scripts/check-jetson-node.sh --host dopey --expected-hostname dopey
```

### 3. Start the control plane

Run the control plane on the development machine or Beelink. Listen on all
interfaces so `dopey` can reach it over the LAN:

```sh
sh scripts/run-control.sh --listen 0.0.0.0:52415 --background
```

Use the development machine or Beelink LAN IP as the control URL from the Jetson,
not `127.0.0.1`.

### 4. Deploy the agent to dopey

```sh
sh scripts/deploy-agent.sh \
  --host dopey \
  --control-url http://<control-host-ip>:52415 \
  --smoke-test
```

The smoke test sends a one-shot heartbeat. It proves the agent binary runs on
`dopey` and can register with the control plane. It does not prove model serving
yet.

### 5. Start a model backend on dopey

Start a real OpenAI-compatible backend on `dopey`, such as `llama-server`, bound
to `127.0.0.1:8080` on the Jetson.

For the POC, prefer a small quantized GGUF model such as
`qwen2.5-coder-1.5b-q4`.

### 6. Start the long-running agent proxy on dopey

Run the agent in proxy mode so the control plane can route requests through the
agent to the node-local model backend:

```sh
ssh dopey '/usr/local/bin/jetsonfabric-agent \
  --control-url http://<control-host-ip>:52415 \
  --join-token dev-token \
  --node-name dopey \
  --listen 0.0.0.0:52416 \
  --advertise-url http://dopey.local:52416 \
  --llama-url http://127.0.0.1:8080 \
  --model qwen2.5-coder-1.5b-q4'
```

### 7. Run the POC smoke test

From the control/development machine:

```sh
sh scripts/poc-dopey-smoke.sh \
  --control-url http://127.0.0.1:52415
```

The smoke test verifies control health, agent health, `dopey` registration,
route preview validity, chat completion output, route metadata, and benchmark
JSONL presence.

## Agent Container

The agent can be built as a small container image, but the preferred POC deploy
path is the native Go binary under host/systemd control. Containerizing the agent
can be revisited after the host telemetry and runtime-management path is stable.

```sh
docker build -f Dockerfile.agent -t jetsonfabric-agent:dev .
```

For Jetson targets, build an arm64 image with Docker Buildx:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.agent -t jetsonfabric-agent:arm64 .
```

## Expanding To More Jetsons

Eventually a new Jetson should be added by installing the agent and giving it the
control-plane URL plus a join token.

Expected join flow:

```bash
./jetsonfabric-agent \
  --control-url http://beelink:52415 \
  --join-token <token> \
  --node-name sleepy
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
7. A second Jetson added later to prove the P0/MVP layer-split path.

## Build

Runtime services are Go-native:

```sh
sh scripts/build.sh
```

This produces Linux development binaries and Linux arm64 binaries for Jetson.
See [docs/build.md](docs/build.md).
