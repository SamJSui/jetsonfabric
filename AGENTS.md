# Agent Instructions

This is the source repo for JetsonFabric. Keep this file intentionally small so coding agents get only the rules needed to work safely in the codebase.

Long-term project memory, design journals, experiments, and ChatGPT context packs belong in the separate private knowledge-base repo: `SamJSui/jetsonfabric-kb`.

## Project Boundary

JetsonFabric is an exo-inspired distributed inference runtime for low-cost Jetson-class edge clusters. The product story is Jetson-first edge inference orchestration: node discovery, model placement, routing, benchmarking, failover, thermal/resource awareness, and an OpenAI-compatible API surface.

Do not turn this repo into a generic homelab dashboard, repo-ingestion chatbot, or single-node local chatbot.

## Implementation Rules

- Production control-plane and agent code should stay in Go.
- Runtime-sensitive inference paths should be C++ first when they become real: TensorRT/ONNX wrappers, llama.cpp integration, activation/tensor transport, layer-shard execution, pinned-buffer experiments, and transport optimization.
- Python is allowed for benchmark analysis, graphing, reports, and notebooks only.
- CUDA belongs only where required for runtime work: pinned or mapped buffers, CPU/GPU transfer measurement, TensorRT or llama.cpp GPU integration, activation compression, or measured GPUDirect/RDMA experiments.
- Do not make Kubernetes mandatory before the scheduler and benchmark loop are useful.
- Do not overstate distributed inference performance; benchmark claims before presenting them as wins.

## Current Priority

Finish the POC before starting real layer-split implementation:

1. Register one Jetson agent.
2. Detect useful Jetson hardware/runtime facts.
3. Route a prompt through `jetsonfabric-control` to the Jetson agent proxy.
4. Have the agent proxy request a real node-local model backend.
5. Return a model response through the control-plane API.
6. Record benchmark data: latency, throughput, memory, thermal, and failures.

After the POC is real and benchmarked, P0/MVP becomes real layer-split inference across Jetson nodes.

## Required Checks

Use the WSL/Linux shell workflow by default:

```sh
go fmt ./...
go test ./...
sh scripts/build.sh
```

For docs-only edits, at least run:

```sh
git diff --check
```

## Source Of Truth

- Project/product context: `docs/project-context.md`
- Engineering bar: `docs/engineering-standards.md`
- Roadmap: `docs/roadmap.md`
- POC plan: `docs/poc-single-node-serving.md`
- P0 layer-split plan: `docs/p0-layer-split-mvp.md`
- KB boundary and external context plan: `docs/knowledge-base-boundary.md`

Preserve user changes. Do not reset or revert unrelated work. Keep commits small and explain what changed.
