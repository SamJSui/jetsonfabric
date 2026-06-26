# Engineering Standards

JetsonFabric should be written like a serious AI infrastructure project, not a
throwaway homelab script. The repo is early, but the quality bar is high because
the project is meant to demonstrate distributed systems judgment, edge AI
pragmatism, and production-minded engineering.

## Default Bar

Every implementation change should satisfy these standards:

- Clear behavior: the change has a specific user, node, API, or runtime outcome.
- Small scope: the change does not mix unrelated refactors with feature work.
- Explicit contracts: inputs, outputs, config, errors, and timeouts are defined.
- Tests: important behavior is covered with unit, handler, integration, or smoke
  tests appropriate to the risk.
- Verification: the final response records the command used to verify the
  change.
- Observability: distributed decisions should be visible through route metadata,
  logs, metrics, benchmark records, or API state.
- Honest claims: performance claims require benchmark evidence.

## Go Standards

Go owns the control plane, node agent, scheduler, route planner, API gateway,
model registry, runtime clients, and benchmark recording path.

Required practices:

- Keep packages cohesive and boring. Prefer small explicit packages over clever
  abstractions.
- Pass `context.Context` through request, backend, and network paths.
- Put timeouts on outbound network calls.
- Return errors with useful context; do not hide the failing node, model, route,
  or backend.
- Avoid `panic` in server and agent paths except for truly unrecoverable startup
  configuration errors.
- Avoid package-level mutable state unless it is deliberately owned by a server
  or registry type.
- Use interfaces at boundaries that need fakes: runtime backends, benchmark
  recorders, clocks, and node stores.
- Keep JSON API types stable and tested.
- Validate config before serving traffic.
- Prefer table-driven tests for scheduler, routing, compatibility, and parsing
  logic.

Minimum implementation verification:

```powershell
$env:GOCACHE='C:\Users\sui\Documents\JetsonFabric\.cache\go-build'
C:\Users\sui\Documents\tools\go\bin\go.exe fmt ./...
C:\Users\sui\Documents\tools\go\bin\go.exe test ./...
.\scripts\build.ps1
```

## C++ And CUDA Standards

C++ should be introduced only for runtime-sensitive inference paths. Do not move
ordinary control-plane logic into C++.

Expected C++ scope:

- TensorRT or ONNX runtime adapters
- llama.cpp integration
- tensor and activation transport
- layer-shard execution
- pinned-buffer and CUDA transfer experiments
- activation compression

Required practices when C++ arrives:

- Prefer RAII and value ownership over manual lifetime management.
- Avoid raw owning pointers.
- Make tensor shape, dtype, byte length, session ID, and step explicit in
  transport headers.
- Check CUDA, TensorRT, and system-call return values.
- Keep CPU/GPU transfer behavior measurable.
- Use profiling data before adding custom CUDA kernels.
- Keep custom kernels small, isolated, and benchmarked against the existing
  runtime.

## Python Boundary

Python is allowed for benchmark analysis, plotting, notebooks, and reports.
Python must not own production serving, node registration, scheduling, model
placement, agent heartbeats, or runtime transport.

If Python scripts are added, they should read recorded benchmark artifacts rather
than becoming the source of truth for cluster behavior.

## API And Runtime Standards

JetsonFabric should expose boring, debuggable APIs.

- Validate request bodies and model IDs.
- Return clear 4xx errors for bad input and clear 5xx/502/503 errors for backend
  and routing failures.
- Include route metadata where useful: route mode, node ID, model ID, backend,
  and benchmark reference.
- Keep OpenAI-compatible endpoints compatible unless an extension is documented.
- Do not leak join tokens, local credentials, or machine-specific secrets in logs
  or benchmark files.

## Scheduling And Benchmark Standards

The scheduler should be evidence-driven.

- Single-node serving is the first baseline.
- Replica mode is a control baseline, not the product identity.
- Layer split starts only after the single-Jetson path is real and benchmarked.
- Tensor parallelism is experimental until network and runtime measurements prove
  it can help.
- Every performance comparison should name the model, quantization, prompt set,
  hardware, network, route mode, and measured metrics.

Required benchmark fields should include, when available:

- model ID
- node ID
- route mode
- backend
- prompt or prompt-set ID
- latency
- time-to-first-token
- output tokens
- tokens/sec
- memory
- temperature
- throttling or power mode
- error state

## Security And Operations

Even local clusters deserve clean operational boundaries.

- Join tokens should be enforced before unknown agents become schedulable.
- Agent-reported capabilities should be treated as input, not blindly trusted.
- Shell commands must not be assembled from untrusted request fields.
- Runtime process management should have explicit paths, arguments, and logs.
- Local config files should be examples unless they intentionally represent a
  checked-in default.
- Secrets and host-specific files must stay out of Git.

## Definition Of Done

A code change is done when:

1. The behavior is implemented in the intended layer.
2. The API/config/runtime contract is clear.
3. Errors are explicit and actionable.
4. Tests cover the main success and failure paths.
5. The standard verification command passes.
6. Docs or examples are updated if the workflow changed.
7. Any shortcut is documented as temporary with a follow-up path.

If a change cannot meet this bar yet, keep it isolated behind an experiment,
feature flag, or clearly documented stub instead of letting it become the main
path.
