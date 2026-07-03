# Single-Node Runtime Prep

This document defines the short hardening pass before JetsonFabric begins the
custom single-node runtime path.

The current POC proves:

```text
jetsonfabric-control -> dopey agent -> llama-server -> model response
```

The next runtime milestone should prove:

```text
jetsonfabric-control -> dopey agent -> jetsonfabric-runtime-worker -> model response
```

with one node, one stage, and all model layers local.

## Why Not Start Runtime Immediately

Before adding a custom runtime, make the proven POC repeatable and reduce known
friction around deployment, process management, and benchmark schema. Otherwise,
runtime bugs will be mixed with service/network/deployment bugs.

## Pre-Runtime Checklist

### 1. Service repeatability

Create a repeatable way to run the three POC processes:

- `jetsonfabric-control` on WSL/dev machine or Beelink.
- `llama-server` on `dopey`.
- `jetsonfabric-agent` on `dopey`.

For near-term development, scripts are acceptable. Before P0/MVP, prefer systemd
units on Jetson for `llama-server` and `jetsonfabric-agent`.

### 2. Redeploy patched agent

Redeploy the current source agent so `dopey` has fixes made after the first POC:

- deploy script no longer treats `--help` as a failed install check;
- missing model artifact catalog no longer blocks runtime startup.

### 3. Benchmark schema cleanup

Normalize benchmark records before serious comparisons. Older desktop records use
`node_id`; real `dopey` records use `node_name`.

Going forward, use `node_name` consistently.

### 4. Runtime contract document

Define the first `jetsonfabric-runtime-worker` contract before writing code:

- startup flags;
- health endpoint;
- full-model single-node chat endpoint;
- future stage endpoint;
- model/shard metadata;
- route metadata emitted or returned to the agent/control path.

### 5. Minimal C++ build lane

Add the smallest possible C++ build target before adding model execution:

- builds on WSL/dev if possible;
- cross-build story for Jetson or native build on Jetson;
- produces `jetsonfabric-runtime-worker`;
- supports `--help` and `/healthz` first.

### 6. Runtime backend registration shape

Add or reserve a backend kind for the custom runtime, such as:

```text
jetsonfabric-runtime
```

The first custom runtime route should still be `single_node` until there is a real
multi-stage activation boundary.

## First Runtime Milestone

The first custom runtime milestone is not distributed inference.

It is:

1. Build `jetsonfabric-runtime-worker`.
2. Run it on `dopey`.
3. Expose a health endpoint.
4. Expose a minimal full-model/single-stage inference endpoint or stub.
5. Register it through the agent as a backend.
6. Route a request through control to this backend.
7. Record a benchmark with `backend_kind=jetsonfabric-runtime`.

Only after this should the runtime introduce explicit layer ranges or tensor
transport.

## Non-Goals Before Single-Node Runtime

- Real layer split.
- Tensor parallelism.
- RDMA or custom NIC work.
- General scheduler redesign.
- Replacing llama-server baseline.

The llama-server path remains the baseline for comparison.
