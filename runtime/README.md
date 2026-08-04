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
peer activations use binary StageWire requests either through node API gateways
or through the direct runtime transport. Multi-stage runtime workers require
the same `JETSONFABRIC_CLUSTER_TOKEN` as their supervising nodes so peer
StageWire calls can authenticate.

Before a managed native JFM load, the runtime inspects the selected package
segments without faulting their tensor pages and compares exact stage weights,
full-context KV and inter-stage activation capacity for every configured
parallel session, and 256 MiB of required post-load headroom with Linux
`MemAvailable`. Native
deployment preparation then allocates that session pool before reporting
`ready`, including worst-case full-context prefill and decode scheduler buffers.
An unsafe replacement is rejected with
`deployment_memory_admission_rejected` before CUDA allocation starts. Engines
must explicitly register estimated or best-effort admission; the check is a
guard against obvious overlap failures. Backend allocation during preparation
remains authoritative for fragmentation and allocator costs that cannot be
predicted exactly.

Native F32 stages keep activations in one byte-backed owned buffer. Ownership
moves from native execution through stage orchestration, and direct HTTP
transport sends the StageWire prefix and activation payload as separate socket
segments without flattening them into another frame. Incoming HTTP parsing and
StageWire decoding still materialize receive storage, so this is not an
end-to-end zero-copy or pinned-buffer path.

## Experimental Tensor-Parallel Runtime

The CUDA build includes an experimental llama.cpp tensor-parallel execution
mode. One driver runtime presents a single logical generation stage while
llama.cpp shards model tensors across its local CUDA device and one or more
remote CUDA devices exposed by `jetsonfabric-tensor-worker`.

Start the provider on a trusted private network:

```bash
make run-tensor-worker \
  TENSOR_WORKER_LISTEN=0.0.0.0:52520 \
  TENSOR_WORKER_ALLOW_REMOTE=true
```

Then start the driver with a provider address and optional local/remote split:

```bash
make run-runtime \
  RUNTIME_MODE=tensor_parallel \
  RUNTIME_COMPUTE_BACKEND=cuda \
  RUNTIME_TENSOR_RPC_PEERS=node-b.local:52520 \
  RUNTIME_TENSOR_SPLIT=3,2 \
  STAGE_INDEX=0 STAGE_COUNT=1 \
  LAYER_START=0 LAYER_END=48
```

The tensor worker uses llama.cpp's raw GGML RPC protocol, which provides no
authentication or encryption. Non-loopback listeners therefore require the
explicit `--allow-remote` acknowledgement and must remain on a trusted LAN.
This first implementation is a direct-runtime research path; coordinator
placement and lifecycle management remain pipeline-only until tensor execution
has a secure transport and a stable rank-assignment contract.

## Tensor-Parallel Feasibility Benchmark

The optional CUDA microbenchmark measures a Qwen-shaped SwiGLU MLP locally and
with its intermediate width split across two Jetsons:

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

The full 14B runtime comparison is documented in
[`docs/benchmarks/2026-08-01-tensor-parallel-runtime.md`](../docs/benchmarks/2026-08-01-tensor-parallel-runtime.md).

## Native Engine Foundation

`engines/native` contains a standalone C model core and the JFM stage-native
package reader. `jf-model-compile` imports GGUF tensors into reusable per-layer
segments without changing their quantized bytes. This path currently measures
model packaging, exact stage selection, and NVMe first-touch; it is not registered as a serving
engine until it passes end-to-end Qwen correctness gates. See
[`docs/native-engine.md`](../docs/native-engine.md).

## Layout

- `worker/`: process entrypoint and validated runtime configuration;
- `api/`: health, deployment lifecycle, generation, and binary stage endpoints;
- `deployment/`: resident deployment state and lifecycle barriers;
- `engine/`: runtime service and engine construction;
- `inference/`: engine-neutral execution interface and request values;
- `adapters/`: llama.cpp full-model and partial-layer execution;
- `engines/native/`: dependency-light C model package and stage-loading core;
- `compiler/`: offline GGUF-to-JFM import tools;
- `tensor_parallel/`: device-mesh validation and trusted-LAN RPC provider;
- `protocol/`: generation, stage, and lifecycle serialization;
- `transport/`: runtime-initiated peer StageWire HTTP transport;
- `benchmarks/`: isolated transport and CUDA feasibility tools;
- `speculative/`: pluggable draft strategies for verified speculative decode;
- `patches/`: the pinned llama.cpp stage-range extension.

See `docs/runtime-stage-interface.md` and `docs/llama-cpp-partial-layer.md` for
the public contracts and current limitations.
