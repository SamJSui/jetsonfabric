# Architecture Diagrams

These SVG diagrams are the visual source of truth for JetsonFabric. They are divided into current implementation and target architecture so future design is not mistaken for merged behavior.

Only diagrams listed here and in [`diagrams/README.md`](diagrams/README.md) are maintained.

## Current implementation

### Codebase map

![JetsonFabric current codebase map](diagrams/codebase-map.svg)

### Go package dependency view

Higher-level packages own policy; lower-level packages own transport and execution mechanisms.

![JetsonFabric Go package dependency diagram](diagrams/package-dependency.svg)

### Type and method contract view

![JetsonFabric current type and method contracts](diagrams/type-contract-view.svg)

### Current service component view

A client may call any Go node. The elected coordinator coordinates the request through peer node APIs and each node's supervised C++ runtime.

![JetsonFabric current service component view](diagrams/component-view.svg)

### Startup sequence

![JetsonFabric node startup sequence](diagrams/startup-sequence.svg)

### Deployment and topology

Logical node count and physical host count are separate. Runtime URLs remain local; peer traffic uses node API URLs.

![JetsonFabric deployment view](diagrams/deployment-view.svg)

### Test strategy

![JetsonFabric test strategy view](diagrams/test-strategy-view.svg)

## Generation ownership

### Runtime-owned call stack

Go selects and admits an immutable plan, then makes one generation call. The
stage-0 C++ runtime owns prefill, decode, peer stage transport, cancellation,
and cleanup.

![JetsonFabric generation call ownership](diagrams/generation-call-stack.svg)

### Current one-call generation sequence

The coordinator selects a prepared plan and calls stage 0 as the runtime
pipeline leader. Token events stream back while the runtime owns prefill,
decode, activation transport, cancellation, and session cleanup.

![JetsonFabric target one-call generation sequence](diagrams/layer-split-sequence.svg)

## Target dynamic runtime architecture

These diagrams describe roadmap intent, not current behavior.

### Dynamic model deployment

![JetsonFabric target dynamic runtime component view](diagrams/future-layer-split-component.svg)

### Model artifact and memory lifecycle

![JetsonFabric target model artifact and memory flow](diagrams/model-artifact-flow.svg)

### Safe rebalance

![JetsonFabric safe pipeline rebalance sequence](diagrams/rebalance-sequence.svg)
