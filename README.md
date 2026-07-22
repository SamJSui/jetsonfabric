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
stage-local weight residency, runtime-owned generation, and automatic
epoch-based rebalance with session drain. Physical multi-Jetson CUDA acceptance
is still required before claiming usable aggregate device memory or a
distributed performance win.

This repository is ready to publish as an experimental source release: the
architecture, limitations, and validation gates are explicit, and CI exercises
the real CPU inference path. It is not production-ready. Installation is still
source-based, cluster traffic assumes a trusted LAN, and physical multi-Jetson
CUDA validation remains open.

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

## Quick start

### Requirements

- Linux development host or NVIDIA Jetson;
- Go toolchain specified by `go.mod`;
- CMake and a C++20 compiler;
- `curl`, `jq`, and standard build utilities;
- a compatible GGUF model;
- CUDA toolkit for GPU builds on Jetson.

The normal quick start runs one Go node supervising one C++ runtime. It uses the
same stage contract as a distributed deployment, but it does not claim to prove
multi-device execution.

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

### Build and run one node

```sh
make setup
make test
make dev-up
```

`make dev-up` runs one full-model pipeline stage using the same stage execution
path used by multi-node deployments. By default, the node listens on `19180`
and its supervised runtime listens on `19190`.

After the build and one-token readiness inference succeed, output ends with a
block like this. PIDs and model details vary by machine:

```text
JetsonFabric development node is ready.

Model:       qwen2.5-coder-1.5b-q4
Layers:      [0,28)
Pipeline:    stage 0 of 1
Backend:     cpu
Node:        http://127.0.0.1:19180
Runtime:     http://127.0.0.1:19190
Node PID:    21452
Runtime PID: 21463
Logs:        /path/to/jetsonfabric/.cache/jetsonfabric/dev/logs
```

In another terminal:

```sh
make dev-status
make dev-chat DEV_PROMPT='Explain JetsonFabric in one sentence.' DEV_MAX_TOKENS=16
make kill
```

Representative `make dev-status` output, abbreviated to the fields most useful
for a first run:

```text
Node: http://127.0.0.1:19180
Runtime: http://127.0.0.1:19190
Node PID: 21452
Runtime PID: 21463

Health:
{"leader":{"node_name":"jf-dev",...},"node_id":"...","service":"jetsonfabric-node","status":"ok"}

Members:
{
  "leader": {"node_name": "jf-dev", "api_url": "http://127.0.0.1:19180", ...},
  "members": [{"node_name": "jf-dev", "runtime_url": "http://127.0.0.1:19190", ...}]
}

Route preview:
{
  "model": "qwen2.5-coder-1.5b-q4",
  "valid": true,
  "mode": "data_parallel",
  "topology": "colocated",
  "stage_count": 1,
  "stages": [{"stage_index": 0, "layer_start": 0, "layer_end": 28, ...}]
}
```

`make dev-chat` prints the OpenAI-compatible response body:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1780000000,
  "model": "qwen2.5-coder-1.5b-q4",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "JetsonFabric ..."},
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28}
}
```

The generated ID, timestamp, text, finish reason, and token counts are not fixed
test values.

### Join two Jetsons

For the first physical cluster, build the CUDA runtime on each Jetson, place the
same GGUF at the same absolute path, and install the same model registry and
cluster token on both nodes:

```sh
make node-linux-arm64
make runtime-cuda RUNTIME_CUDA_ARCH=87
sh scripts/install-node-layout.sh
```

Start both nodes with `--runtime-idle`; the elected coordinator will load and
activate the assigned layer range when a deployment is switched. A complete
trusted-LAN command and model-registry example are in the
[node join guide](docs/node-join.md).

Once both nodes appear in membership, switch a two-stage deployment through
either node:

```sh
curl -sS http://dopey.local:52415/v1/cluster/members | jq

curl -sS -X POST http://dopey.local:52415/v1/deployments/switch \
  -H 'Content-Type: application/json' \
  --data-binary @examples/deployment-switch-request.json | jq

curl -sS http://dopey.local:52415/v1/deployments/active | jq

curl -sS -X POST http://grumpy.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json | jq
```

Requests sent to a non-coordinator node are forwarded through the node facade.
A successful switch reports an active deployment epoch with two ordered stages:

```json
{
  "phase": "active",
  "active": {
    "deployment_id": "qwen-two-node",
    "epoch": 1,
    "model": {
      "model_id": "qwen2.5-coder-1.5b-q4",
      "execution_mode": "pipeline_parallel",
      "layer_count": 28
    },
    "stages": [
      {"stage_index": 0, "stage_count": 2, "node_name": "dopey", "layer_start": 0, "layer_end": 14},
      {"stage_index": 1, "stage_count": 2, "node_name": "grumpy", "layer_start": 14, "layer_end": 28}
    ]
  },
  "compatibility": {
    "architecture": "arm64",
    "runtime_revision": "dev",
    "llama_cpp_revision": "dev",
    "compute_backend": "cuda",
    "cuda_active": true
  }
}
```

Node IDs, epoch values, assignments, and response fields omitted with `...` in
these examples vary with the live cluster.

### Configuration and examples

Keep the existing directories; they represent different boundaries:

| Path | Purpose |
| --- | --- |
| `.env.example` | Template for one developer machine; copy it to the ignored `.env.local`. |
| `configs/models.example.json` | Tracked model metadata used for planning and as a registry schema example. A deployable registry must also contain the local artifact path and SHA-256. |
| `examples/*.json` | Executable request bodies for `curl` and the benchmark client. |

Do not commit GGUF files, tokens, host-specific paths, generated deployment
plans, or mutable node state. Those belong under ignored local paths or the
system directories described in the node join guide.

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

### Milestone 0 - Distributed inference foundation

- [x] Peer discovery, membership, and coordinator election.
- [x] Pipeline topology planning and layer-range assignment.
- [x] Supervised node-local runtime workers.
- [x] Versioned binary Stagewire activation protocol.
- [x] Real partial-layer `llama.cpp` execution.
- [x] Persistent prefill/decode state and sampled-token equivalence tests.
- [x] OpenAI-compatible non-streaming chat completion path.

**Outcome:** JetsonFabric can execute a real model as an ordered pipeline and
prove activation continuity and token equivalence in integration tests.

### Milestone 1 - Runtime model ownership

- [x] Introduce `ModelManager` as the owner of active model execution state.
- [x] Route model identity, stage execution, and session cleanup through it.
- [x] Preserve existing behavior with native and real-model regression tests.

**Outcome:** runtime service lifetime is no longer structurally coupled to raw
engine and stage-worker ownership.

### Milestone 2 - Dynamic deployment lifecycle

- [x] Allow `ModelManager` to have no active deployment.
- [x] Define deployment identity, assignment, and lifecycle states.
- [x] Add `load`, `activate`, `drain`, `unload`, and `status` operations.
- [x] Allow the runtime process to start idle without loading a model.
- [x] Reject inference clearly when no deployment is active.
- [x] Add an idle -> load -> activate -> infer -> drain -> unload integration test.

**Outcome:** a long-lived runtime can change the model stage it hosts without a
process restart.

### Milestone 3 - Versioned cluster deployment plans

- [x] Convert member capabilities and model metadata into a deployment plan.
- [x] Assign a versioned deployment epoch to every stage.
- [x] Enforce engine, model ID, artifact hash, architecture, layer count, runtime
  revision, and `llama.cpp` revision compatibility.
- [x] Add CUDA-build and GPU-execution attestation to placement metadata.
- [x] Apply a deployment plan manually before enabling automatic reconciliation.

**Outcome:** the coordinator can prepare a consistent, auditable deployment
across multiple runtimes.

### Milestone 4 - True model-memory partitioning

- [x] Load only tensors required by a stage's assigned layer range.
- [x] Separate on-disk model registration from in-memory model residency.
- [x] Add partition memory accounting, deployment pinning, and explicit eviction.
- [x] Verify that each stage consumes less weight memory than a full-model runtime.
- [x] Prove with real model weights that both partitions fit a per-device weight
  capacity that excludes the full model.

**Outcome:** stage-local model-weight residency is partitioned and accounted for.
Physical per-process and CUDA memory evidence remains part of the hardware
acceptance milestone before claiming aggregate usable device memory.

### Milestone 5 - Runtime-owned generation data plane

- [x] Replace coordinator-driven per-stage calls with one runtime `Generate` call.
- [x] Select deterministic stage 0 as the runtime pipeline-session leader.
- [x] Move prefill, repeated decode passes, stream-boundary cancellation, and cleanup into the
  runtime data plane.
- [x] Send activations between runtime workers without relaying every
  stage payload through the Go coordinator.
- [x] Add streaming output with bounded synchronous backpressure.

**Outcome:** Go owns control-plane policy while runtimes own the latency-sensitive
distributed generation loop.

### Milestone 6 - Automatic reconciliation and safe rebalance

- [x] Recompute desired placement when membership or capacity changes.
- [x] Prepare every partition of a new deployment epoch before activation.
- [x] Route new sessions to the new epoch while old sessions remain pinned.
- [x] Drain old sessions before unloading obsolete partitions.
- [x] Add rollback, timeout, node-loss, and partial-prepare recovery.

**Outcome:** JetsonFabric can add, remove, or lose a Jetson without interrupting
healthy active generations or restarting the cluster.

### Milestone 7 - Physical multi-Jetson CUDA acceptance

- [ ] Run the pipeline on at least two distinct physical Jetsons.
- [ ] Prove CUDA execution rather than configuration-only CUDA claims.
- [ ] Require baseline token equivalence and activation CRC continuity.
- [ ] Capture per-stage latency, transfer bytes, throughput, memory, utilization,
  power, temperature, and throttling.
- [ ] Validate startup, shutdown, reconnect, and failure behavior on real hardware.

**Outcome:** physical evidence supports the distributed CUDA inference claim.

### Milestone 8 - Operational readiness

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
- Safe replacement temporarily keeps old and new partitions resident on reused
  nodes. A rebalance can fail cleanly when a device lacks that overlap capacity;
  measured per-stage overlap estimation is not implemented yet.
- Reconciliation intent and deployment state are leader-local until durable
  coordinator state moves to a consensus store. Coordinator failover does not
  reconstruct an active deployment automatically yet.
- A session whose assigned runtime is lost cannot continue. The controller
  preserves healthy sessions, publishes a replacement when capacity exists, and
  retries cleanup if an unreachable runtime later returns.
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
- [Node join flow](docs/node-join.md)
- [Deployment standards](docs/deployment-standards.md)
- [Deployment invariants](docs/deployment-invariants.md)
- [Runtime stage interface](docs/runtime-stage-interface.md)
- [Partial-layer llama.cpp integration](docs/llama-cpp-partial-layer.md)
- [Physical Jetson validation](docs/physical-jetson-validation.md)
