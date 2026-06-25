# Agent Context

This repo is JetsonMesh: an exo-inspired distributed inference runtime for
low-cost Jetson-class edge clusters.

## Project Intent

Build a Go/C++ native edge AI fabric that makes multiple Jetson Orin-class
devices behave like one observable inference cluster. The first credible target
is not frontier-model replacement. The target is measurable edge-specific value:
placement, routing, failover, cost-aware serving, thermal awareness, and
evidence about when distributed inference helps or loses.

Interview pitch:

> I built JetsonMesh, an exo-inspired distributed inference runtime for
> low-cost Jetson edge clusters. It auto-discovers Jetson nodes, profiles
> compute/network/thermal behavior, places model work across devices, exposes an
> OpenAI-compatible API, and benchmarks when layer-split inference beats or
> loses to single-node execution.

## Current Stack

- Go control plane in `cmd/jetsonmesh-control` and `internal/control`.
- Go node agent in `cmd/jetsonmesh-agent` and `internal/agent`.
- Shared cluster types in `internal/cluster`.
- Model registry in `internal/modelregistry`.
- Placement preview logic in `internal/routing`.
- Platform/runtime detection in `internal/system`.
- Example model config in `configs/models.example.json`.
- PowerShell scripts in `scripts/` for local build and dev runs.

Python is allowed for benchmark analysis, graphing, reports, and notebooks only.
Do not move the production control plane, node agent, scheduler, or runtime
transport into Python.

C++ belongs in later runtime-sensitive paths:

- TensorRT/ONNX execution wrappers
- llama.cpp integration
- activation/tensor transport
- layer-shard execution
- pinned-buffer and transfer optimization experiments

## Hardware Framing

- Beelink Mini S13: optional control plane, dashboard host, storage, and
  development node.
- Jetson Orin Nano / Orin Nano Super: primary AI compute worker class.
- Raspberry Pi 5: optional future sentinel, sensor, or compatibility node. It is
  not the core performance story for model inference.

The cleanest product story is Jetson-first. Avoid turning the repo into a
generic homelab dashboard or a single-node local chatbot.

## Non-Goals And Honesty Boundaries

- Do not claim JetsonMesh beats frontier models.
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
2. Benchmark each node against known model backends.
3. Plan routes for single-node, replica baseline, layer-split, and fallback
   modes.
4. Make route decisions observable with latency, throughput, memory, thermal,
   power, network bytes/token, and failure data.
5. Implement layer-split inference first as a clear distributed systems story.
6. Keep tensor parallelism as an experimental path because network
   synchronization can dominate on ordinary Ethernet.

## Verification Commands

Use the local Go toolchain unless the system `go` is known to be correct:

```powershell
$env:GOCACHE='C:\Users\sui\Documents\JetsonMesh\.cache\go-build'
C:\Users\sui\Documents\tools\go\bin\go.exe fmt ./...
C:\Users\sui\Documents\tools\go\bin\go.exe test ./...
.\scripts\build.ps1
```

For docs-only edits, at least run:

```powershell
git diff --check
```

## Git And Repo State

The GitHub repo is private:

```text
https://github.com/SamJSui/JetsonMesh
```

Preserve user changes. Do not reset or revert unrelated work. Keep commits small
and explain what changed.

## Next Useful Work

- Add dev resource overrides so Windows smoke tests can preview plausible
  Jetson memory/accelerator data.
- Add authenticated join-token enforcement.
- Add a benchmark result data model and persistence.
- Add Jetson-specific detection for JetPack, CUDA, TensorRT, power mode,
  temperature, throttling, and `tegrastats`.
- Implement a dashboard/API surface that mirrors exo-style node and route
  visibility.
- Add a real single-Jetson model backend before attempting layer split.
