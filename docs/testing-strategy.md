# Testing Strategy

JetsonFabric should not chase unit tests for every file as a checkbox. It should maintain focused tests around stable contracts, risky logic, and regression-prone boundaries.

## Testing Pyramid

1. Pure unit tests for deterministic logic.
2. HTTP handler tests for API behavior.
3. Contract tests for runtime/backend boundaries.
4. Smoke tests for real process and hardware paths.
5. Benchmark tests as evidence, not pass/fail unit tests.

## Must-Have Go Unit Tests

### `internal/routing`

Test placement decisions:

- unknown model
- no nodes
- enough memory
- insufficient memory
- preferred accelerator present/missing
- deterministic placement ordering

### `internal/layersplit`

Test planner and transport helper behavior:

- one-node single-stage mode when supported by future runtime contract
- two-node split layer ranges
- weighted layer allocation
- invalid layer counts
- invalid candidate sets
- stage role assignment
- activation request/response normalization

### `internal/modelregistry` and `internal/modelartifacts`

Test config parsing:

- valid catalog/registry
- missing model ID
- missing runtime
- invalid source URL
- pipeline_parallel model without enough layers
- artifact lookup by model ID

### `internal/runtimeclient`

Use `httptest.Server`:

- successful OpenAI-compatible response
- backend HTTP error response
- malformed JSON
- response with no choices
- usage-derived token stats
- timeout/error propagation

### `internal/control`

Use handler tests with fake backends/recorders:

- health endpoint
- heartbeat registration
- unauthorized heartbeat
- nodes endpoint ordering
- route preview
- chat completion success
- chat completion no compatible node
- benchmark recorder called with expected fields
- route metadata includes node/backend/latency

### `internal/agent`

Use fake control and runtime servers:

- heartbeat payload includes system snapshot fields
- join token header is sent
- proxy forwards chat requests
- proxy strips route metadata from backend response
- runtime unavailable path

### `internal/system`

Keep tests pure where possible:

- meminfo parsing
- load average parsing
- field helpers

Do not over-test live host command availability because it depends on the machine running tests.

## Runtime Tests

When the C++ runtime lane begins:

### Unit tests

- CLI/config parsing
- tensor frame encode/decode
- dtype enum serialization
- shape validation
- byte-length validation
- transport frame boundaries
- checksum/length mismatch handling

### Contract tests

- `/healthz`
- future `/v1/runtime/info`
- single-node chat/stub endpoint
- future stage endpoint

### Hardware tests

Run only on Jetson or explicitly marked hardware environments:

- CUDA availability
- model load
- one-node runtime generation
- thermal/memory sampling

## Smoke Tests

Keep smoke tests as scripts because they validate real processes:

- `scripts/poc-dopey-smoke.sh`
- future `scripts/runtime-single-node-smoke.sh`
- future `scripts/layer-split-two-node-smoke.sh`

Smoke tests should assert route metadata and benchmark output, not only that an HTTP response exists.

## Near-Term Test Priorities

Before heavy runtime work:

1. Add routing tests.
2. Add control handler tests for heartbeat and chat route success.
3. Add runtimeclient tests with `httptest.Server`.
4. Add layersplit planner tests.
5. Add model registry/artifact parser tests.
6. Add a benchmark schema test or fixture check so old `node_id` vs new `node_name` drift does not continue.

## Rule

Every new module should have tests if it owns either:

- route decisions
- serialized formats
- benchmark fields
- network/API behavior
- runtime boundaries
- hardware detection parsing

Simple glue files and scripts do not need exhaustive unit tests, but they should be covered by smoke tests when they are part of the POC path.
