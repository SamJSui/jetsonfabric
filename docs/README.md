# JetsonFabric Documentation

The repository README is the project overview, getting-started guide, and public
roadmap. Files under `docs/` provide current implementation contracts and
validation procedures.

## Start Here

- [Local development](local-development.md) - configure, run, inspect, test, and
  stop a local node and runtime.
- [Node join flow](node-join.md) - bootstrap a Jetson through mDNS or static
  seeds.
- [Single-node runtime validation](single-node-runtime-validation.md) - build and
  verify the first real runtime path.
- [Architecture](architecture.md) - current component ownership and request flow.
- [Architecture diagrams](architecture-diagrams.md) - curated current and target
  architecture views.
- [Deployment invariants](deployment-invariants.md) - current and target
  constraints for model admission, residency, epochs, membership, and sessions.

## Runtime Contracts

- [Runtime stage interface](runtime-stage-interface.md) - engine-neutral stage
  input, output, identity, and lifecycle boundary.
- [Stagewire v1](stagewire-v1.md) - binary inter-stage frame and payload contract.
- [llama.cpp partial-layer execution](llama-cpp-partial-layer.md) - pinned engine
  integration and stage-range behavior.

## Validation

- [Testing strategy](testing-strategy.md) - required CI, integration, and
  hardware gates.
- [Physical two-Jetson CUDA validation](physical-jetson-validation.md) - hardware
  acceptance gate for distributed CUDA execution.

Real-model local integration commands are documented in
[Local development](local-development.md). The source repository does not keep
separate milestone-snapshot guides for completed phases.

## Documentation Rules

- Describe merged behavior as current implementation.
- Label future behavior explicitly as target architecture.
- Keep one canonical page per contract or workflow.
- Put project history, design rationale, experiment logs, and hardware journals
  in `SamJSui/jetsonfabric-kb`.
- Use Git history and merged pull requests for superseded milestone snapshots.
- Verify commands, ports, APIs, and file paths against the current source tree.
