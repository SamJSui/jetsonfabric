# JetsonFabric

Distributed pipeline-parallel LLM inference for clusters of NVIDIA Jetson devices.

JetsonFabric is an experimental edge-inference fabric that makes a group of
Jetson-class machines behave like one logical inference system. Every machine
runs the same Go node process, which discovers peers, maintains membership,
participates in coordinator election, plans pipeline stages, exposes an
OpenAI-compatible API, and supervises a node-local C++ runtime worker.

The runtime worker hosts inference-engine adapters such as `llama.cpp` and
executes the transformer-layer range assigned to that device.

## Project goal

The end goal is a self-organizing Jetson cluster that can:

- discover available devices and their capabilities;
- divide one model into ordered pipeline stages;
- load only the model weights owned by each stage;
- use aggregate cluster memory for models that do not fit on one device;
- serve requests through an OpenAI-compatible endpoint;
- rebalance safely when devices join, leave, or fail;
- preserve active generation sessions while a new deployment becomes active.

JetsonFabric currently proves real partial-layer execution and activation
handoff. It does **not** yet pool model-weight memory across devices.

## Features

### Implemented

- identical `jetsonfabric-node` process on every machine;
- static and mDNS peer discovery;
- fresh membership views and deterministic coordinator election;
- topology-aware pipeline planning and layer-range arithmetic;
- node-local supervision of the C++ `jetsonfabric-runtime-worker`;
- real `llama.cpp` partial-layer execution for `llama` and `qwen2` graphs;
- binary F32 activation transport through the versioned Stagewire protocol;
- persistent stage-local prefill and decode state;
- coordinator-owned autoregressive token generation and session cleanup;
- OpenAI-compatible non-streaming `/v1/chat/completions` routing through any node;
- model artifact hashing and runtime capability advertisement;
- `ModelManager` ownership of the runtime's active model execution components;
- real-model single-stage and colocated pipeline integration tests;
- synthetic binary-activation transport tests and native C++ unit tests.

### Current inference path

```text
OpenAI-compatible client
  -> any jetsonfabric-node
  -> elected coordinator
  -> ordered stage plan
  -> stage node facade
  -> node-local runtime worker
  -> assigned transformer layers
  -> activation forwarded to the next stage
  -> final-stage logits and sampled token
```

For a two-stage generation:

```text
prefill
  prompt -> stage 0 -> activation -> stage 1 -> sampled token

decode
  token  -> stage 0 -> activation -> stage 1 -> sampled token
```

All stages retain the same generated session ID across prefill and decode.
Every individual stage operation receives its own request ID for tracing.

## Getting started

### Requirements

- Linux development host or NVIDIA Jetson;
- Go toolchain specified by `go.mod`;
- CMake and a C++20 compiler;
- `curl`, `jq`, and standard build utilities;
- a compatible GGUF model;
- CUDA toolkit for GPU builds on Jetson.

### Configure

```sh
git clone https://github.com/SamJSui/jetsonfabric.git
cd jetsonfabric
cp .env.example .env.local
```

Edit `.env.local` and set at minimum:

```dotenv
MODEL=qwen2.5-coder-1.5b-q4
MODEL_PATH=/absolute/path/to/model.gguf
```

For a CPU-only development machine:

```dotenv
RUNTIME_COMPUTE_BACKEND=cpu
RUNTIME_N_GPU_LAYERS=0
```

### Run one development node

```sh
make setup
make test
make dev-up
```

`make dev-up` runs one full-model pipeline stage using the same stage execution
path used by multi-node deployments. By default, the node listens on `19180`
and its supervised runtime listens on `19190`.

In another terminal:

```sh
make dev-status
make dev-chat DEV_PROMPT='Explain JetsonFabric in one sentence.' DEV_MAX_TOKENS=16
make kill
```

### Build explicitly

CPU runtime:

```sh
make runtime
```

CUDA runtime for Jetson Orin:

```sh
make runtime-cuda RUNTIME_CUDA_ARCH=87
```

Node binaries:

```sh
make node
```

### Run integration tests

Single-stage real-model integration:

```sh
make test-integration-single \
  MODEL_PATH=/absolute/path/to/model.gguf \
  MODEL=qwen2.5-coder-1.5b-q4
```

Two-stage colocated real-model integration:

```sh
make test-integration-pipeline \
  MODEL_PATH=/absolute/path/to/model.gguf \
  MODEL=qwen2.5-coder-1.5b-q4
```

The colocated test uses these default port pairs:

```text
stage 0: node 19180, runtime 19190
stage 1: node 19181, runtime 19191
```

## Roadmap

Checkboxes represent merged and tested behavior. Each remaining milestone should
be delivered through small, reviewable PRs with focused acceptance criteria.

### Milestone 0 — Distributed inference foundation

- [x] Peer discovery, membership, and coordinator election.
- [x] Pipeline topology planning and layer-range assignment.
- [x] Supervised node-local runtime workers.
- [x] Versioned binary Stagewire activation protocol.
- [x] Real partial-layer `llama.cpp` execution.
- [x] Persistent prefill/decode state and sampled-token equivalence tests.
- [x] OpenAI-compatible non-streaming chat completion path.

**Outcome:** JetsonFabric can execute a real model as an ordered pipeline and
prove activation continuity and token equivalence in integration tests.

### Milestone 1 — Runtime model ownership

- [x] Introduce `ModelManager` as the owner of active model execution state.
- [x] Route model identity, stage execution, and session cleanup through it.
- [x] Preserve existing behavior with native and real-model regression tests.

**Outcome:** runtime service lifetime is no longer structurally coupled to raw
engine and stage-worker ownership.

### Milestone 2 — Dynamic deployment lifecycle

- [ ] Allow `ModelManager` to have no active deployment.
- [ ] Define deployment identity, assignment, and lifecycle states.
- [ ] Add `prepare`, `activate`, `unload`, and `status` operations.
- [ ] Allow the runtime process to start idle without loading a model.
- [ ] Reject inference clearly when no deployment is active.
- [ ] Add an idle -> prepare -> activate -> infer -> unload integration test.

**Outcome:** a long-lived runtime can change the model stage it hosts without a
process restart.

### Milestone 3 — Versioned cluster deployment plans

- [ ] Convert member capabilities and model metadata into a deployment plan.
- [ ] Assign a versioned deployment epoch to every stage.
- [ ] Enforce engine, model ID, artifact hash, architecture, layer count, runtime
  revision, and `llama.cpp` revision compatibility.
- [ ] Add CUDA-build and GPU-execution attestation to placement metadata.
- [ ] Apply a deployment plan manually before enabling automatic reconciliation.

**Outcome:** the coordinator can prepare a consistent, auditable deployment
across multiple runtimes.

### Milestone 4 — True model-memory partitioning

- [ ] Load only tensors required by a stage's assigned layer range.
- [ ] Separate on-disk model registration from in-memory model residency.
- [ ] Add partition memory accounting, pinning, and eviction.
- [ ] Verify that each stage consumes less memory than a full-model runtime.
- [ ] Run a model whose total weights exceed the memory of any single device.

**Outcome:** the cluster pools aggregate device memory, not only compute.

### Milestone 5 — Runtime-owned generation data plane

- [ ] Replace coordinator-driven per-stage calls with one runtime `Generate` call.
- [ ] Elect or select a runtime pipeline-session leader.
- [ ] Move prefill, repeated decode passes, cancellation, and cleanup into the
  runtime data plane.
- [ ] Send activations directly between runtime workers instead of relaying every
  stage payload through the Go coordinator.
- [ ] Add streaming output and backpressure after the one-call path is stable.

**Outcome:** Go owns control-plane policy while runtimes own the latency-sensitive
distributed generation loop.

### Milestone 6 — Automatic reconciliation and safe rebalance

- [ ] Recompute desired placement when membership or capacity changes.
- [ ] Prepare every partition of a new deployment epoch before activation.
- [ ] Route new sessions to the new epoch while old sessions remain pinned.
- [ ] Drain old sessions before unloading obsolete partitions.
- [ ] Add rollback, timeout, node-loss, and partial-prepare recovery.

**Outcome:** JetsonFabric can add, remove, or lose a Jetson without interrupting
healthy active generations or restarting the cluster.

### Milestone 7 — Physical multi-Jetson CUDA acceptance

- [ ] Run the pipeline on at least two distinct physical Jetsons.
- [ ] Prove CUDA execution rather than configuration-only CUDA claims.
- [ ] Require baseline token equivalence and activation CRC continuity.
- [ ] Capture per-stage latency, transfer bytes, throughput, memory, utilization,
  power, temperature, and throttling.
- [ ] Validate startup, shutdown, reconnect, and failure behavior on real hardware.

**Outcome:** physical evidence supports the distributed CUDA inference claim.

### Milestone 8 — Operational readiness

- [ ] Package repeatable installation and service management for Jetson devices.
- [ ] Add authentication and secure cluster admission.
- [ ] Add structured metrics, traces, logs, and deployment inspection tools.
- [ ] Add model-management commands and operator documentation.
- [ ] Expand OpenAI API compatibility after the core distributed runtime is stable.

**End goal:** a practical, observable, self-reconfiguring edge inference fabric
that combines the compute and memory of multiple Jetson devices.

## Current limitations

- Transformer execution is partitioned, but every runtime still opens the full
  GGUF; model-weight memory is not partitioned yet.
- The Go coordinator currently owns both the token loop and stage loop.
- Runtime assignments are still established at startup rather than through a
  dynamic deployment lifecycle.
- The generation path is sequential and does not overlap microbatches.
- Chat completions are non-streaming and use greedy sampling.
- Physical multi-Jetson CUDA execution has not yet completed the acceptance gate.

## Documentation

- [Architecture diagrams](docs/architecture-diagrams.md)
- [Local development](docs/local-development.md)
- [Runtime stage interface](docs/runtime-stage-interface.md)
- [Partial-layer llama.cpp integration](docs/llama-cpp-partial-layer.md)
- [Physical Jetson validation](docs/physical-jetson-validation.md)
