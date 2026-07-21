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

JetsonFabric currently proves real partial-layer execution, partitioned
stage-local weight residency, activation handoff, and runtime-owned generation.
Physical multi-Jetson CUDA acceptance is still required before claiming usable
aggregate device memory or a distributed performance win.

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
- runtime-owned autoregressive token generation, cancellation at stream
  boundaries, and cleanup;
- one coordinator-to-runtime generation call per completion;
- runtime-initiated peer Stagewire transport that bypasses the coordinator;
- OpenAI-compatible buffered and streaming `/v1/chat/completions` through any node;
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
  -> one authenticated Generate call to the stage-0 node
  -> stage-0 runtime GenerationRunner
  -> assigned transformer layers on stage 0
  -> peer node facade and node-local runtime for each later stage
  -> Stagewire activation forwarded by the runtime data plane
  -> final-stage logits and sampled token
  -> NDJSON token events translated to a buffered response or SSE
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

- [x] Allow `ModelManager` to have no active deployment.
- [x] Define deployment identity, assignment, and lifecycle states.
- [x] Add `load`, `activate`, `unload`, and `status` operations.
- [x] Allow the runtime process to start idle without loading a model.
- [x] Reject inference clearly when no deployment is active.
- [x] Add an idle -> load -> activate -> infer -> unload integration test.

**Outcome:** a long-lived runtime can change the model stage it hosts without a
process restart.

### Milestone 3 — Versioned cluster deployment plans

- [x] Convert member capabilities and model metadata into a deployment plan.
- [x] Assign a versioned deployment epoch to every stage.
- [x] Enforce engine, model ID, artifact hash, architecture, layer count, runtime
  revision, and `llama.cpp` revision compatibility.
- [x] Add CUDA-build and GPU-execution attestation to placement metadata.
- [x] Apply a deployment plan manually before enabling automatic reconciliation.

**Outcome:** the coordinator can prepare a consistent, auditable deployment
across multiple runtimes.

### Milestone 4 — True model-memory partitioning

- [x] Load only tensors required by a stage's assigned layer range.
- [x] Separate on-disk model registration from in-memory model residency.
- [x] Add partition memory accounting, deployment pinning, and explicit eviction.
- [x] Verify that each stage consumes less weight memory than a full-model runtime.
- [x] Prove with real model weights that both partitions fit a per-device weight
  capacity that excludes the full model.

**Outcome:** stage-local model-weight residency is partitioned and accounted for.
Physical per-process and CUDA memory evidence remains part of the hardware
acceptance milestone before claiming aggregate usable device memory.

### Milestone 5 — Runtime-owned generation data plane

- [x] Replace coordinator-driven per-stage calls with one runtime `Generate` call.
- [x] Select deterministic stage 0 as the runtime pipeline-session leader.
- [x] Move prefill, repeated decode passes, stream-boundary cancellation, and cleanup into the
  runtime data plane.
- [x] Send activations between runtime workers without relaying every
  stage payload through the Go coordinator.
- [x] Add streaming output with bounded synchronous backpressure.

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
- [ ] Replace shared-token bootstrap with per-node identity, TLS, and secure
  cluster admission.
- [ ] Add structured metrics, traces, logs, and deployment inspection tools.
- [ ] Add model-management commands and operator documentation.
- [ ] Expand OpenAI API compatibility after the core distributed runtime is stable.

**End goal:** a practical, observable, self-reconfiguring edge inference fabric
that combines the compute and memory of multiple Jetson devices.

## Current limitations

- Stage-local tensor residency currently supports Llama and Qwen2 models through
  a patch tied to the pinned `llama.cpp` revision.
- `model_memory` reports tensor payload bytes. Allocator overhead, compute
  buffers, and context/KV structures are not included in that number yet.
- Placement conservatively applies `min_memory_gb` to every selected stage until
  the planner has a measured per-stage memory estimator.
- Deployment switching is explicit and destructive; automatic reconciliation,
  rollback, and spare-node preparation remain future milestones.
- The runtime generation path is sequential, uses one HTTP connection per remote
  stage operation, and does not overlap microbatches.
- Client cancellation is observed between stage passes or stream writes; an
  in-flight peer request remains bounded by its transport timeout.
- Chat completions support buffered responses and SSE but use greedy sampling.
- Stage data-plane requests require the shared cluster token, but traffic is
  plaintext HTTP without TLS or per-node identity. Use only a trusted network.
- Physical multi-Jetson CUDA execution has not yet completed the acceptance gate.

## Documentation

- [Architecture diagrams](docs/architecture-diagrams.md)
- [Local development](docs/local-development.md)
- [Runtime stage interface](docs/runtime-stage-interface.md)
- [Partial-layer llama.cpp integration](docs/llama-cpp-partial-layer.md)
- [Physical Jetson validation](docs/physical-jetson-validation.md)
