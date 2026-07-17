# Diagram Set

The SVG files in this directory are the checked-in visual source of truth for
JetsonFabric. `../architecture-diagrams.md` is the curated index.

## Current implementation

- `codebase-map.svg` — repository and service/module map.
- `package-dependency.svg` — current Go package dependency direction.
- `type-contract-view.svg` — current Go structs, C++ classes, and key methods.
- `component-view.svg` — current any-node service and stage path.
- `startup-sequence.svg` — current node/runtime startup lifecycle.
- `deployment-view.svg` — logical nodes versus physical hosts.
- `test-strategy-view.svg` — automated proof layers.

## Target architecture

- `generation-call-stack.svg` — current Go-owned loops versus the target runtime-owned generation loop.
- `layer-split-sequence.svg` — one external runtime `Generate` call with internal prefill/decode passes.
- `future-layer-split-component.svg` — dynamic `ModelManager`, versioned deployments, and direct runtime data plane.
- `model-artifact-flow.svg` — artifact catalog, in-memory partition cache, session pins, and eviction.
- `rebalance-sequence.svg` — prepare, activate, drain, and retire a new deployment epoch.

Target diagrams are design intent, not claims about the current implementation.
