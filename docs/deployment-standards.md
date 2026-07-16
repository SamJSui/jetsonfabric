# Deployment Standards

JetsonFabric should be written as if this repository may eventually become public. A new developer should be able to clone the repo, follow documented setup steps, build the project, and run a useful local or small-cluster configuration without needing private context from `jetsonfabric-kb` or prior chat history.

## Public-Ready Baseline

Every deployable feature should satisfy these standards before it is treated as a real milestone:

- The default documented path builds from a fresh clone.
- Required system packages, toolchain versions, CUDA assumptions, and hardware assumptions are documented.
- Optional dependencies are explicit and opt-in.
- Local paths, hostnames, Tailscale IPs, model paths, and Jetson names are examples, not hard-coded requirements.
- The failure mode for a missing optional dependency is actionable.
- Build commands work without private files, private repos, or unpublished context.
- Runtime commands have copy-pasteable examples.
- Any required model artifact is documented as user-provided and is not committed to the repo.
- Generated build artifacts, third-party checkouts, caches, downloaded models, and local data directories are gitignored.

## Build Standards

The top-level `Makefile` is the public entrypoint. It should orchestrate Go checks, node builds, runtime builds, and developer utilities without hiding critical flags.

`make build` should remain the standard full-project check for source changes that affect product code.

The common runtime paths should be short and memorable:

```sh
make runtime             # default C++ runtime build
make setup-llama-cpp     # clone llama.cpp if missing
make runtime-llama-cpu   # opt-in llama.cpp engine build without CUDA
make runtime-llama-cuda  # opt-in llama.cpp engine build with CUDA
```

Runtime C++ dependencies belong under `runtime/CMakeLists.txt`. Optional engine integrations such as llama.cpp must be controlled by explicit CMake options, but most developers should not have to type the raw CMake flags directly.

CUDA builds must be explicit, because not every developer machine has CUDA. The public Makefile target should expose simple knobs instead of requiring a long command:

```sh
make runtime-llama-cuda RUNTIME_BUILD_JOBS=1 RUNTIME_CUDA_ARCH=87 CUDA_NVCC=/usr/local/cuda/bin/nvcc
```

Jetson builds can require constrained parallelism. The build should expose a job-count knob instead of assuming unlimited memory.

## Dependency Standards

Small source dependencies may use CMake `FetchContent` when they are pinned and reliable for public builds. Header-only or small libraries such as `nlohmann/json` are acceptable.

Large engine dependencies such as llama.cpp should default to a local, gitignored checkout or an explicitly configured path. Do not commit large third-party engine source trees or model files into this repo.

Expected layout:

```text
runtime/
  third_party/
    llama.cpp/      # local checkout, gitignored
```

The build must fail with a clear message if an optional dependency is enabled but missing.

## Runtime Standards

A public user should be able to run one node and one runtime locally before attempting a cluster.

Minimum documented runtime path:

```text
jetsonfabric-node
  -> jetsonfabric-runtime-worker
      -> inference engine adapter
      -> compute backend such as CPU or CUDA
```

Use project terminology consistently:

- `runtime`: JetsonFabric-owned host process that accepts node-routed execution work and owns engine adapters.
- `engine`: inference implementation such as llama.cpp or TensorRT-LLM.
- `compute_backend`: execution API such as CPU or CUDA.
- `adapter`: JetsonFabric integration wrapper around an engine.
- `executor`: runtime component that executes a full-model request or an assigned layer range.

Do not present synthetic payload transforms as product functionality. Synthetic behavior is allowed only for narrow tests or temporary seams that unblock real engine integration.

## Configuration Standards

Configuration should support home-lab defaults while remaining portable:

- `dopey`, `beehive`, and Tailscale IPs may appear in examples only.
- Defaults should work on a single Linux machine where possible.
- mDNS should be optional and paired with static seed fallback.
- WSL should be treated as a development client unless explicitly run as a node.
- Paths under `.cache/`, `runtime/build/`, `dist/`, `models/`, and `runtime/third_party/` must not be required to exist in a fresh clone unless documented setup commands create them.

## Documentation Standards

For each real deployment capability, document:

1. prerequisites;
2. dependency setup;
3. build command;
4. run command;
5. health check;
6. expected success output;
7. common failures and fixes.

Keep private reasoning, long-form project memory, experiment journals, and hardware notes in `SamJSui/jetsonfabric-kb`. Keep source-facing deployment instructions in this repo.

## Release Readiness Bar

Before treating a feature as public-ready:

```sh
gofmt -w ./cmd ./internal ./tools
go test ./...
make build
```

For runtime/CUDA changes, also run the relevant runtime build and document any hardware-specific command used.

If a feature requires hardware that CI or a generic developer machine cannot provide, the CPU/non-accelerated path should still build, and the accelerated path should be clearly opt-in.
