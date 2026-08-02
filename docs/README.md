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
- [Codebase catalog](architecture/codebase-catalog.md) - every Go package,
  behavior-owning type, C++ runtime class, and their relationships.
- [Architecture diagrams](architecture-diagrams.md) - curated current and target
  architecture views.
- [Deployment invariants](deployment-invariants.md) - current and target
  constraints for model admission, residency, epochs, membership, and sessions.
- [Deployment standards](deployment-standards.md) - public source-build,
  configuration, security, and release expectations.

## Runtime Contracts

- [Runtime stage interface](runtime-stage-interface.md) - engine-neutral stage
  input, output, identity, and lifecycle boundary.
- [Stagewire v2](stagewire-v2.md) - binary inter-stage frame, rollback, and payload contract.
- [llama.cpp partial-layer execution](llama-cpp-partial-layer.md) - pinned engine
  integration and stage-range behavior.
- [Native engine](native-engine.md) - JFM model packages, exact stage selection,
  build commands, and serving correctness gates.
- [JFM v2 binary format](jfm-v2-format.md) - byte layout, tensor-type ABI,
  topology, validation, and trust boundary.

## Validation

- [Testing strategy](testing-strategy.md) - required CI, integration, and
  hardware gates.
- [Physical two-Jetson CUDA validation](physical-jetson-validation.md) - hardware
  acceptance gate for distributed CUDA execution.
- [Two-Orin Nano benchmark](benchmarks/2026-07-27-two-orin-nano.md) - physical
  correctness, capacity, performance, power, and HumanEval results.
- [Concurrent serving optimization](benchmarks/2026-07-28-concurrent-serving.md)
  - two-node Qwen2.5-Coder 1.5B aggregate throughput, TTFT, ITL, latency,
  measured stage balance, and the matching one-node baseline.
- [Serving matrix](benchmarks/2026-07-29-serving-matrix.md) - replica and
  balanced-pipeline results for Qwen2.5-Coder 1.5B, 3B, 7B, and 14B, including
  runtime timing decomposition and HumanEval useful-work rates.
- [32B capacity frontier](benchmarks/2026-07-30-32b-capacity.md) - stage-local
  Q8_0 KV memory, balanced 33/31 serving, sustained concurrency, and the
  largest Qwen2.5-Coder quantization tested on the two-node cluster.
- [Tensor-parallel feasibility](benchmarks/2026-07-31-tensor-parallel-feasibility.md)
  - wired-network lower bound and a two-rank CUDA SwiGLU MLP sublayer proof for
  Qwen2.5-Coder 1.5B and 14B shapes.
- [Tensor-parallel runtime](benchmarks/2026-08-01-tensor-parallel-runtime.md) -
  full-model 14B CUDA tensor sharding compared with the matched two-node
  pipeline on the same Jetsons and Gigabit Ethernet.
- [Native engine foundation](benchmarks/2026-08-01-native-engine-foundation.md)
  - 14B JFM import, exact stage selection, NVMe first-touch optimization, and
  explicit native-serving limitations.
- [Native KV-cache benchmark](benchmarks/2026-08-01-native-kv-cache.md) -
  matched llama.cpp token parity, full-prefix ablation, direct-engine latency,
  and CUDA transfer evidence.

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
