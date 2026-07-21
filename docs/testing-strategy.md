# Testing Strategy

JetsonFabric tests protect the contracts that can corrupt model execution,
publish an invalid deployment, or misrepresent hardware evidence. The suite uses
real model execution where engine behavior matters and focused fakes where
failure injection matters.

## Required Pull Request Gates

GitHub Actions runs these gates on Linux:

1. `go test ./...` for node fabric, planning, lifecycle, facade, and wire logic.
2. Linux ARM64 cross-compilation of `jetsonfabric-node`.
3. Shell syntax validation for every script under `scripts/`.
4. Native C++ runtime tests against a pinned llama.cpp revision.
5. Single-stage, lifecycle, model-switch, and colocated pipeline tests with
   immutable GGUF fixtures.
6. Qwen2 partition-residency and token-equivalence coverage.
7. Synthetic binary Stagewire transport coverage.

Model fixtures must use immutable source revisions and exact SHA-256 checks. A
cache hit never replaces integrity verification.

## Go Coverage

The highest-value Go tests cover:

- `internal/discovery` and `internal/membership`: peer normalization, freshness,
  duplicate handling, and coordinator election inputs;
- `internal/clusterplan`: deterministic placement, physical-host separation,
  compatibility, contiguous layer ranges, and immutable deployment epochs;
- `internal/coordinator`: epoch-scoped admission, ready and activation barriers,
  membership/capacity reconciliation, partial failure and timeout rollback,
  node-loss cleanup retry, session pinning, one-call generation, runtime event
  accounting, and incremental SSE flushing;
- `internal/facade`: public API routing, fail-closed peer authentication, and
  node-local runtime forwarding;
- `internal/stagewire`: frame bounds, metadata validation, payload length, and
  checksum handling, including managed deployment identity;
- `internal/runtimebridge`: runtime lifecycle and streaming generation gateway
  contracts, including credential stripping at the local runtime boundary.

Use `httptest.Server` for HTTP boundaries. Use fakes only to inject states that
are difficult to reproduce deterministically with a native runtime, and pair
important success paths with real-process integration tests.

## Native Runtime Coverage

Native tests own engine-sensitive behavior:

- runtime configuration and deployment lifecycle;
- exact partial-layer residency ranges;
- one-, two-, and three-stage llama.cpp execution;
- real middle-stage activation forwarding;
- greedy token equivalence through repeated decode steps;
- session retention, ordering, expiry, and cleanup;
- runtime-owned prefill/decode loops, cancellation cleanup, and stage-call
  accounting;
- EOS exclusion, natural-stop pass accounting, cleanup-response identity, and
  strict Stagewire media-type parsing;
- supported architecture checks for llama and qwen2.

The full-model baseline and partitioned stages currently share the pinned
llama.cpp build. Immutable fixture hashes and checked-in expected tokens should
be used for critical fixtures so a shared regression cannot redefine the
expected result silently.

## Integration Scripts

Current real-process harnesses are:

- `scripts/local/validate-single-node.sh`;
- `scripts/local/validate-runtime-lifecycle.sh`;
- `scripts/local/validate-coordinator-model-switch.sh`;
- `scripts/local/validate-colocated-pipeline.sh`;
- `scripts/jetson/validate-distributed-cuda.sh`.

Integration scripts must validate response semantics, the complete deployment
identity (deployment ID, epoch, model ID, and artifact SHA-256), stage count,
activation byte and CRC continuity, and exact greedy tokens. A successful HTTP
status alone is not sufficient.

The single-stage harness proves the one-call local path. The colocated pipeline
harness calls `/v1/runtime/generate` directly, requires its token IDs to equal
the full-model baseline, proves authenticated remote stage calls, and requires
buffered and SSE chat text and finish reasons to match. Stage-call counts must
also account for the hidden EOS pass when the public response stops naturally.
The model-switch harness proves managed deployment identity, overlapping model
replacement, stale-epoch rejection, and automatic one-stage recovery after a
worker is stopped. CI invokes the colocated harness with the pinned Qwen2
fixture as an additional natural-stop case across the direct runtime, buffered
chat, and SSE chat paths so EOS is excluded while its final stage pass remains
accounted for.

## Hardware Acceptance

Physical CUDA acceptance is separate from portable CI. It requires at least two
distinct Jetson hosts selected into one distributed plan. Every selected member
must report CUDA as the configured backend and explicitly attest that CUDA is
active. Capture `tegrastats` with stage traces so GPU utilization, memory,
temperature, activation size, and latency are tied to the same run.

Colocated stages on one Jetson prove orchestration and engine correctness. They
do not satisfy the physical distributed-compute gate.

## Change Rule

Add or update tests whenever a change owns routing, deployment lifecycle,
serialized data, model residency, network behavior, runtime execution, or
hardware claims. Keep narrow glue code covered through the closest integration
harness rather than adding low-value tests for every line.
