# JetsonFabric Runtime

`jetsonfabric-runtime-worker` is the node-local C++ inference process. The Go
node owns discovery, membership, deployment policy, facade routing, and the
public API. The runtime owns model residency, llama.cpp execution, generation
loops, peer stage forwarding, session state, and the latency-sensitive stage
boundary.

## Build

CPU:

```bash
make runtime
```

CUDA on Jetson Orin:

```bash
make runtime-cuda RUNTIME_CUDA_ARCH=87 RUNTIME_BUILD_JOBS=1
```

Both targets prepare the pinned llama.cpp revision and verify that the
JetsonFabric stage-range extension applies.

## Run

Normally `jetsonfabric-node` supervises the runtime with `--runtime-url auto`.
For direct lifecycle work:

```bash
./dist/jetsonfabric-runtime-worker \
  --listen 127.0.0.1:9090 \
  --idle \
  --engine llama.cpp \
  --compute-backend cpu \
  --mode pipeline_parallel
```

The coordinator then uses the runtime lifecycle endpoints to load, activate,
inspect, drain, and unload exact deployment epochs. A runtime can keep an old
draining epoch resident beside a replacement epoch during safe handoff. Generation enters through
`POST /v1/generate` on the stage-0 runtime as newline-delimited JSON events;
peer activations use binary Stagewire requests either through node API gateways
or through the direct runtime transport. Multi-stage runtime workers require
the same `JETSONFABRIC_CLUSTER_TOKEN` as their supervising nodes so peer
Stagewire calls can authenticate.

## Tensor-Parallel Feasibility Benchmark

The optional CUDA benchmark measures a Qwen-shaped SwiGLU MLP locally and with
its intermediate width split across two Jetsons. It is a research tool, not a
runtime execution mode:

```bash
cmake -S runtime -B runtime/build-tp \
  -DCMAKE_BUILD_TYPE=Release \
  -DJF_BUILD_CUDA_BENCHMARKS=ON \
  -DJF_CUDA_ARCHITECTURES=87
cmake --build runtime/build-tp \
  --target jetsonfabric-tensor-parallel-mlp-bench -j2
```

See
[`docs/benchmarks/2026-07-31-tensor-parallel-feasibility.md`](../docs/benchmarks/2026-07-31-tensor-parallel-feasibility.md)
for commands, methodology, results, and claim boundaries.

## Layout

- `worker/`: process entrypoint and validated runtime configuration;
- `api/`: health, deployment lifecycle, generation, and binary stage endpoints;
- `deployment/`: resident deployment state and lifecycle barriers;
- `engine/`: runtime service and engine construction;
- `adapters/`: llama.cpp full-model and partial-layer execution;
- `protocol/`: generation, stage, and lifecycle serialization;
- `transport/`: runtime-initiated peer Stagewire HTTP transport;
- `benchmarks/`: isolated transport and CUDA feasibility tools;
- `speculative/`: pluggable draft strategies for verified speculative decode;
- `patches/`: the pinned llama.cpp stage-range extension.

See `docs/runtime-stage-interface.md` and `docs/llama-cpp-partial-layer.md` for
the public contracts and current limitations.
