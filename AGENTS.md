# Agent Context

This repo is JetsonFabric: an exo-inspired distributed inference runtime for
low-cost Jetson-class edge clusters.

## Project Intent

Build a Go/C++ native edge AI fabric that makes multiple Jetson Orin-class
devices behave like one observable inference cluster. The first credible target
is not frontier-model replacement. The target is measurable edge-specific value:
placement, routing, failover, cost-aware serving, thermal awareness, and
evidence about when distributed inference helps or loses.

Interview pitch:

> I built JetsonFabric, an exo-inspired distributed inference runtime for
> low-cost Jetson edge clusters. It auto-discovers Jetson nodes, profiles
> compute/network/thermal behavior, places model work across devices, exposes an
> OpenAI-compatible API, and benchmarks when layer-split inference beats or
> loses to single-node execution.

## Current Stack

- Go control plane in `cmd/jetsonfabric-control` and `internal/control`.
- Go node agent in `cmd/jetsonfabric-agent` and `internal/agent`.
- Shared cluster types in `internal/cluster`.
- Model registry in `internal/modelregistry`.
- Model artifact catalog in `internal/modelartifacts` and
  `configs/model-artifacts.example.json`.
- Placement preview logic in `internal/routing`.
- Platform/runtime detection in `internal/system`.
- Example model config in `configs/models.example.json`.
- POSIX `sh` scripts in `scripts/` for WSL/Linux build and dev runs.
- PowerShell scripts remain only as Windows compatibility helpers.
- Future distributed-runtime work should be C++ first, with CUDA introduced only
  for GPU memory movement, TensorRT/llama.cpp integration, activation
  compression, or measured transfer bottlenecks.

## Engineering Standard

This codebase should be held to a high production-quality bar even while it is
early. Read [docs/engineering-standards.md](docs/engineering-standards.md)
before making implementation changes.

Do not merge work that is only a demo shortcut unless it is isolated, documented,
and intentionally marked as temporary. A change is not complete until it has
clear behavior, tests at the right level, explicit errors, and a verification
command.

Python is allowed for benchmark analysis, graphing, reports, and notebooks only.
Do not move the production control plane, node agent, scheduler, or runtime
transport into Python.

C++ belongs in later runtime-sensitive paths. Prefer C++ over C for JetsonFabric
runtime code because RAII and typed value ownership are a better fit for
long-running tensor transport, CUDA resources, sockets, and runtime sessions.
C APIs are acceptable at external boundaries such as `libibverbs`, POSIX
sockets, CUDA, or runtime libraries, but wrap them in small C++ types instead of
letting raw handles spread through the codebase.

- TensorRT/ONNX execution wrappers
- llama.cpp integration
- activation/tensor transport
- layer-shard execution
- pinned-buffer and transfer optimization experiments
- 10GbE TCP transport experiments
- optional RDMA transport experiments

CUDA belongs only where it is required for runtime work:

- pinned or mapped host buffers
- CPU/GPU transfer measurement
- TensorRT or llama.cpp GPU integration
- activation compression kernels
- possible GPUDirect/RDMA experiments after TCP and 10GbE baselines exist

## Hardware Framing

- Beelink Mini S13: optional control plane, dashboard host, storage, and
  development node.
- Jetson Orin Nano / Orin Nano Super: primary AI compute worker class.
- Raspberry Pi 5: optional future sentinel, sensor, or compatibility node. It is
  not the core performance story for model inference.

The cleanest product story is Jetson-first. Avoid turning the repo into a
generic homelab dashboard or a single-node local chatbot.

## Non-Goals And Honesty Boundaries

- Do not claim JetsonFabric beats frontier models.
- Do not claim distributed inference is always faster.
- Do not make replicated serving the only product identity; keep it as a
  baseline/control mode.
- Do not make Kubernetes mandatory before the scheduler and benchmark loop are
  useful.
- Do not overstate tensor parallelism over Ethernet. Treat it as a stretch
  experiment unless benchmarks prove otherwise.
- Do not spend time on repo/document digestion features; the desired UX is closer
  to Codex/exo-style routing and orchestration.

## Primary Technical Direction

The core research/product lane is profile-driven distributed inference on cheap
edge compute:

1. Discover nodes and collect hardware/runtime/thermal capabilities.
2. Get one real model serving on one Jetson and route one prompt through the
   control plane.
3. Benchmark that single-Jetson backend against known prompts and metrics.
4. Plan routes for single-node, replica_serving, layer_split, and fallback
   modes.
5. Make route decisions observable with latency, throughput, memory, thermal,
   power, network bytes/token, and failure data.
6. Implement layer-split inference only after the single-Jetson serving path is
   real and benchmarked.
7. Optimize the distributed runtime after layer split exists: compare 1GbE TCP,
   10GbE TCP, activation compression, pipelining, and only then RDMA or tensor
   parallelism.
8. Keep tensor parallelism as an experimental path because network
   synchronization can dominate on ordinary Ethernet.

## Current Priority

P0 is single-Jetson model serving. Do not begin layer-split implementation until
JetsonFabric can:

- register one Jetson agent;
- detect useful Jetson hardware/runtime facts;
- route a prompt through `jetsonfabric-control` to the Jetson agent proxy;
- have the agent proxy that request to a real node-local model backend;
- return a model response through the control-plane API;
- record a benchmark result with latency, throughput, memory, and thermal data.

## Verification Commands

Use the WSL/Linux shell workflow by default:

```sh
go fmt ./...
go test ./...
sh scripts/build.sh
```

For implementation changes, prefer the full sequence above. Add narrower tests
only when they support the full check; do not use them as a substitute for final
verification.

For docs-only edits, at least run:

```sh
git diff --check
```

## Git And Repo State

The GitHub repo is private:

```text
https://github.com/SamJSui/jetsonfabric
```

Preserve user changes. Do not reset or revert unrelated work. Keep commits small
and explain what changed.

## Next Useful Work

- Add Jetson-specific detection for JetPack, CUDA, TensorRT, power mode,
  temperature, throttling, and `tegrastats`.
- Expand model artifact management from a catalog into verified download and
  runtime launch lifecycle.
- Add request logging and route-decision logging for control and agent proxy
  calls.
- Add authenticated request handling between control and agent proxy.
- Add dev resource overrides so Windows smoke tests can preview plausible
  Jetson memory/accelerator data.
- Implement a dashboard/API surface that mirrors exo-style node and route
  visibility.
- Add a real single-Jetson model backend before attempting layer split.
- Defer 10GbE, RDMA, GPUDirect, and tensor-parallel work until after P1
  layer-split measurements prove that transport optimization is the bottleneck.
