# Deployment Standards

JetsonFabric is currently distributed as an experimental source release. A
public checkout must be sufficient to build, test, and run the supported local
and small-cluster workflows without private documentation or machine-specific
state.

## Supported Release Posture

- Linux is the supported development and deployment environment.
- NVIDIA Jetson Orin is the CUDA deployment target.
- `jetsonfabric-node` is the only product-facing process. It supervises or
  connects to one node-local `jetsonfabric-runtime-worker`.
- Installation is source-based until the operational-readiness milestone adds
  versioned packages and service units.
- Cluster APIs are suitable only for a trusted network until mutually
  authenticated and encrypted transport is implemented.

## Build and Dependency Rules

- Pin external runtime dependencies, including the exact `llama.cpp` revision.
- Keep optional checkouts and generated build outputs ignored.
- Fail with an actionable message when CUDA, a model artifact, or another
  optional dependency is absent.
- A clean checkout must pass `go test ./...` and build through the documented
  Make targets before release.
- CUDA claims require the physical validation gate; successful configuration or
  cross-compilation alone is insufficient evidence.

## Filesystem and Configuration

The standard device layout is:

```text
/opt/jetsonfabric/                 installed binaries
/etc/jetsonfabric/node.env        private cluster token
/etc/jetsonfabric/models.json     model registry
/var/lib/jetsonfabric/node/       stable node identity
/var/lib/jetsonfabric/models/     local GGUF artifacts
/var/lib/jetsonfabric/benchmarks/ benchmark records
/var/lib/jetsonfabric/logs/       service logs
```

Repository templates and examples must not contain private hostnames, private
addresses, credentials, or local absolute paths. Host-specific registries,
tokens, model files, generated plans, and mutable node state must remain
untracked.

## Cluster Contract

- Every node in one cluster uses the same cluster ID and shared bootstrap token.
- Each node has a stable data directory and a unique persisted node ID.
- Node names are operator labels; they are not security identities.
- Every candidate node must expose compatible architecture, engine, runtime
  revision, `llama.cpp` revision, compute backend, model hash, and layer count.
- A dynamic deployment is published only after every selected runtime has
  loaded and activated its assigned layer range.

## Public Release Checklist

1. Verify the README quick start against a clean checkout.
2. Run the unit, native, integration, shell-syntax, and documentation checks.
3. Confirm model and dependency downloads are pinned and hash-verified in CI.
4. Confirm limitations and incomplete hardware gates are stated without
   performance claims.
5. Confirm no model weights, secrets, private addresses, generated state, or
   private knowledge-base references are required by source-facing workflows.
