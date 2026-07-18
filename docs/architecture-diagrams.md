# Architecture Diagrams

These checked-in SVG diagrams are the visual source of truth for JetsonFabric.
They are divided into **current implementation** and **target architecture** so a
future design is never mistaken for code that already exists.

## Current Implementation

### Codebase Map

A repository-level tour from the Go service entrypoint to the C++ partial-layer
runtime.

![JetsonFabric current codebase map](diagrams/codebase-map.svg)

### Go Package Dependency View

Arrows show caller/importer direction. Higher-level packages own policy; lower-
level packages own transport and execution mechanisms.

![JetsonFabric Go package dependency diagram](diagrams/package-dependency.svg)

### Type and Method Contract View

The main Go structs and C++ classes involved in one distributed generation.

![JetsonFabric current type and method contracts](diagrams/type-contract-view.svg)

### Current Service Component View

A client may call any Go node. The elected leader coordinates the request through
peer node APIs and each node's supervised C++ runtime.

![JetsonFabric current service component view](diagrams/component-view.svg)

### Startup Sequence

The node binds its listener, derives advertised addresses, starts the supervised
runtime, publishes self membership, and begins discovery.

![JetsonFabric node startup sequence](diagrams/startup-sequence.svg)

### Deployment and Topology

Logical node count and physical host count are separate. Runtime URLs remain
local; peer traffic uses node API URLs.

![JetsonFabric deployment view](diagrams/deployment-view.svg)

### Test Strategy

![JetsonFabric test strategy view](diagrams/test-strategy-view.svg)

## Generation Ownership

### Current Versus Target Call Stack

Today Go `stageexec` owns both the token loop and stage loop. The target moves the
generation data plane behind one runtime `Generate` call while preserving the
unavoidable internal prefill/decode passes.

![JetsonFabric generation call ownership](diagrams/generation-call-stack.svg)

### Target One-Call Distributed Generation Sequence

The coordinator selects a prepared plan and calls one runtime pipeline leader.
The runtime leader owns prefill, repeated decode passes, direct activation
transport, cancellation, and session cleanup.

![JetsonFabric target one-call generation sequence](diagrams/layer-split-sequence.svg)

## Target Dynamic Runtime Architecture

These diagrams describe the next milestone. They are design targets, not claims
about the current runtime.

### Dynamic Model Deployment Component View

Runtimes load model partitions on demand from versioned deployment plans. Backend
is placement metadata; model and activation compatibility determine correctness.

![JetsonFabric target dynamic runtime component view](diagrams/future-layer-split-component.svg)

### Model Artifact and In-Memory Lifecycle

A model may be registered and present on disk without occupying runtime memory.
`ModelManager`, `ModelCache`, and `SessionManager` prepare, pin, drain, and evict
stage-local model partitions.

![JetsonFabric target model artifact and memory flow](diagrams/model-artifact-flow.svg)

### Safe Rebalance Sequence

When a new node joins, the coordinator prepares a new deployment epoch, waits for
all partitions to become ready, routes new sessions to the new epoch, lets old
sessions drain on their original plan, and only then unloads obsolete partitions.

![JetsonFabric safe pipeline rebalance sequence](diagrams/rebalance-sequence.svg)

## Supporting Historical Diagrams

The remaining SVGs under `docs/diagrams/` preserve earlier discovery, identity,
and proof-of-concept views. The diagrams linked above supersede them for the
current distributed-runtime architecture.
