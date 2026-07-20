# JetsonFabric Documentation

The repository README is the project overview, getting-started guide, and public roadmap. The files under `docs/` provide current implementation contracts and validation procedures.

## Start here

- [Local development](local-development.md) — configure, run, inspect, test, and stop a local node and runtime.
- [Architecture](architecture.md) — current component ownership and request flow.
- [Architecture diagrams](architecture-diagrams.md) — curated current and target architecture views.
- [Deployment invariants](deployment-invariants.md) — target constraints for model admission, residency, epochs, membership, and session ownership.

## Runtime contracts

- [Runtime stage interface](runtime-stage-interface.md) — engine-neutral stage input, output, identity, and lifecycle boundary.
- [Stagewire v1](stagewire-v1.md) — binary inter-stage frame and payload contract.
- [llama.cpp partial-layer execution](llama-cpp-partial-layer.md) — pinned engine integration and stage-range behavior.

## Validation

- [Physical two-Jetson CUDA validation](physical-jetson-validation.md) — hardware acceptance gate for distributed CUDA execution.

The real-model local integration commands are documented in [Local development](local-development.md). The source repository does not keep separate milestone-snapshot guides for completed phases.

## Documentation rules

- Describe merged behavior as current implementation.
- Label future behavior explicitly as target architecture.
- Keep one canonical page per contract or workflow.
- Put project history, design rationale, experiment logs, and hardware journals in `SamJSui/jetsonfabric-kb`.
- Use Git history and merged pull requests for superseded milestone snapshots.
- Verify commands, ports, APIs, and file paths against the current source tree.
