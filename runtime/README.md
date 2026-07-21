# JetsonFabric Runtime

`jetsonfabric-runtime-worker` is the node-local C++ inference process. The Go
node owns discovery, membership, deployment policy, facade routing, and remote
stage forwarding. The runtime owns model residency, llama.cpp execution,
session state, and the latency-sensitive stage boundary.

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
inspect, and unload an exact deployment epoch.

## Layout

- `worker/`: process entrypoint and validated runtime configuration;
- `api/`: health, deployment lifecycle, chat, and binary stage HTTP endpoints;
- `deployment/`: resident deployment state and lifecycle barriers;
- `engine/`: runtime service and engine construction;
- `adapters/`: llama.cpp full-model and partial-layer execution;
- `protocol/`: stage and lifecycle serialization;
- `patches/`: the pinned llama.cpp stage-range extension.

See `docs/runtime-stage-interface.md` and `docs/llama-cpp-partial-layer.md` for
the public contracts and current limitations.
