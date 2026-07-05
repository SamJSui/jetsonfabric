# Agent Instructions

This is the source repo for JetsonFabric. Keep this file intentionally small so coding agents get only the rules needed to work safely in the codebase.

Long-term project memory, design journals, experiments, hardware notes, troubleshooting narratives, and ChatGPT context packs belong in the separate private knowledge-base repo: `SamJSui/jetsonfabric-kb`.

## Project Boundary

JetsonFabric is a Jetson-first distributed inference fabric for low-cost edge clusters. The current product process is `jetsonfabric-node`: nodes discover peers, maintain membership, elect one coordinator, expose one API surface, and forward runtime work to node-local C++/CUDA runtimes.

Do not reintroduce user-facing `control` or `agent` processes. Historical control-plane logic belongs under `internal/coordinator`; public node API behavior belongs under `internal/facade`; peer state belongs under `internal/membership` and `internal/discovery`.

Do not turn this repo into a generic homelab dashboard, repo-ingestion chatbot, or single-node local chatbot.

## Implementation Rules

- Go owns node fabric, discovery, membership, facade routing, coordinator planning, and tests.
- C++ owns runtime-sensitive inference paths: model execution, layer stages, activation/tensor transport, pinned-buffer experiments, and transport optimization.
- Python is allowed for benchmark analysis, graphing, reports, and notebooks only.
- CUDA belongs only where required for runtime work: pinned or mapped buffers, CPU/GPU transfer measurement, TensorRT or llama.cpp GPU integration, activation compression, or measured GPUDirect/RDMA experiments.
- Do not make Kubernetes mandatory before the scheduler and benchmark loop are useful.
- Do not overstate distributed inference performance; benchmark claims before presenting them as wins.
- Prefer small functions. Aim for 20-40 lines unless the function is simple table setup or unavoidable glue.

## No Stub Milestones

Do not spend project capacity building fake workflows as product milestones. Stubs, fakes, and synthetic executors are allowed only as narrow tests, temporary compile seams, or explicit scaffolding that is immediately replaced by the real runtime path.

When choosing between a convincing demo and a real project step, choose the real project step. For runtime work, prioritize integrating a real model backend and CUDA-capable execution path over synthetic payload transformations. The target architecture is:

```text
jetsonfabric-node
  -> jetsonfabric-runtime-worker
      -> real backend integration, initially llama.cpp/ggml/CUDA where practical
      -> JetsonFabric-owned planning, layer-stage boundaries, activation transport, telemetry, and benchmarking
```

## Current Priority

Keep the node fabric coherent while moving toward real larger-than-one-node model execution:

1. `jetsonfabric-node` is the only product process.
2. Discovery is membership bootstrap, not scheduling truth.
3. Role-gated deterministic election with a local lease/epoch selects the coordinator until there are three coordinator-capable voters for Raft.
4. The coordinator creates deployment or routing decisions.
5. Runtime execution goes through the node facade into the node-local runtime gateway.
6. Build dopey runtime correctness with a real model backend before spending effort on fake pipeline demos.
7. Use one-node `pipeline_parallel` only when it proves the same real runtime contract that later becomes multi-node layer-sharded execution.

## Required Checks

Use the WSL/Linux shell workflow by default:

```sh
gofmt -w ./cmd ./internal ./tools
go test ./...
make build
```

For docs-only edits, at least run:

```sh
git diff --check
```

## Source Of Truth

- Current architecture summary: `README.md`.
- Source-facing workflow and file interaction map: `docs/architecture/node-fabric-workflow.md`.
- Long-term memory and planning: `SamJSui/jetsonfabric-kb`.

Preserve user changes. Do not reset or revert unrelated work. Keep commits small and explain what changed.
