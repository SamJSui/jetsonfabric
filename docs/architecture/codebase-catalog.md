# Codebase Catalog

This page maps the current source tree by ownership. Go has structs and
interfaces rather than classes; the tables below use **component** for a type
that owns behavior and **record** for a type that primarily carries data.

## Process Boundary

Every host runs the same pair of processes:

```text
jetsonfabric-node (Go)
  owns discovery, membership, election, planning, deployment coordination,
  the public API, and the local-runtime facade

jetsonfabric-runtime-worker (C++)
  owns model residency, stage execution, KV-backed generation sessions,
  activation encoding, and latency-sensitive peer transport
```

The Go node may be the elected coordinator, but it is never a separate
`control` process. A remote node always calls another node's public facade; the
node-local runtime address is not a cluster endpoint.

## End-to-End Ownership

```text
client
  -> facade.Router
       -> follower: reverse proxy to elected coordinator
       -> leader: coordinator.Server
            -> DeploymentController or GenerationController
            -> runtimebridge client
                 -> stage-0 node facade
                      -> local C++ HttpServer / RuntimeService
                           -> GenerationService / ModelManager
                           -> GenerationRunner
                                -> local StageWorker
                                -> StageTransport -> peer node facade
                                     -> peer StageWorker
```

There are two deliberate execution paths:

1. `POST /v1/chat/completions` uses C++ `GenerationRunner`. Stage 0 owns the
   prefill/decode loop and sends activations directly to peer runtimes.
2. `POST /v1/layer-split/run` uses Go `stageexec.Executor`. It is a diagnostic
   traversal that records every stage payload, CRC, and latency for validation.

Combining these paths would make the production hot path depend on diagnostic
instrumentation and would return activations through the coordinator again.

## Go Packages

| Package | Responsibility | Important types |
| --- | --- | --- |
| `cmd/jetsonfabric-node` | Parse CLI flags, build `node.Config`, handle process signals, and run the node. | `flagValues` is parsing state; `main`, `run`, and `parseConfig` form the entry path. |
| `internal/api` | Canonical HTTP paths and authentication header names shared by Go components. | Constants only. |
| `internal/benchmarks` | Append coordinator benchmark records without coupling handlers to JSONL files. | `Record`, `Recorder`, `NoopRecorder`, `JSONLRecorder`. |
| `internal/chat` | Engine-neutral OpenAI-compatible response records used across coordinator code. | `CompletionRequest`, `CompletionResponse`, `Message`, `Choice`, `Usage`, `RouteMetadata`, `RouteStage`. |
| `internal/cluster` | Shared cluster vocabulary and model/runtime capability records. | `Engine`, `ExecutionMode`, `DeviceClass`, `OperatingSystem`, `ComputeBackend`, `EngineEndpoint`, `ModelProfile`. |
| `internal/clusterplan` | Pure placement policy, compatibility checks, layer assignment, route previews, and immutable deployment plans. It performs no network I/O. | `Policy`, `Request`, `RoutePreview`, `Placement`, `Stage`, `LayerRange`, `DeploymentIdentity`, `DeploymentModelIdentity`, `DeploymentPlan`, `DeploymentBuildRequest`, `DeploymentBuildResult`, `DeploymentCompatibility`. |
| `internal/config` | Cross-package default filesystem locations. | Constants only. |
| `internal/coordinator` | Leader-only application services and HTTP translation. It owns deployment intent, epoch transitions, reconciliation, generation admission, and public model/deployment/chat APIs. | `Server`, `DeploymentController`, `deploymentState`, `deploymentAdmission`, `GenerationController`, `generationSession`, plus request/response records local to each handler. |
| `internal/discovery` | Find peers with static seeds or mDNS and hydrate lightweight discovery records through HTTP announce. | `Source`, `SelfFunc`, `StaticSource`, `MDNSSource`, `MDNSAdvertiser`, `HydratingSource`, `MultiSource`, `AnnounceClient`, `MDNSConfig`. |
| `internal/election` | Deterministically rank eligible members and maintain a node-local leader lease/epoch. This is not consensus. | `Tracker`, `State`, `Result`, `Candidate`. |
| `internal/facade` | Expose one public API on every node, serve node-local cluster/runtime routes, authenticate peer writes, and proxy coordinator-owned routes to the elected leader. | `Router`, `Config`, `ClusterView`. |
| `internal/inference` | Semantic stage lifecycle rules independent of HTTP and binary framing. | `Phase`, `SessionState`, `Event`, `StagePosition`, `PayloadKind`. |
| `internal/membership` | Define a node record and maintain the thread-safe, freshness-aware local peer view. | `Member`, `NodeRole`, `Store`. |
| `internal/modelartifacts` | Stream a model artifact through SHA-256 without loading the file into memory. | Functions only. |
| `internal/modelregistry` | Load `models.json`, validate model profiles, and resolve a model ID. It stores metadata, not model weights. | `Registry`. |
| `internal/node` | Composition root for one product process: configuration, stable identity, runtime supervision, self-capabilities, HTTP wiring, discovery, and shutdown. | `App`, `Config`, `RuntimeSupervisor`. |
| `internal/runtimebridge` | Go-to-C++ HTTP boundary. Proxies node-local runtime routes and provides typed deployment/generation clients for coordinator calls. | `StageProxy`, `DeploymentProxy`, `GenerationProxy`, `HTTPDeploymentClient`, `HTTPGenerationClient`, their interfaces, and runtime contract records. |
| `internal/stageexec` | Diagnostic Go pipeline traversal and trace collection for `/v1/layer-split/run`. | `Executor`, `Config`, `Request`, `Result`, `StageTrace`, `StageError`. |
| `internal/stagewire` | Go implementation of the versioned binary stage frame. | `Frame`, `Metadata`, `DeploymentIdentity`, `Operation`; stage request/response aliases. |
| `internal/system` | Detect hostname, OS, architecture, Jetson/CUDA/container capabilities, memory, load, and metrics. | `Snapshot`. |
| `tools/bench` | External benchmark client that measures request latency, TTFT, ITL, throughput, errors, and streamed token accounting. | `Summary`, `LatencySummary`, `RequestResult`. |

## Go Components

### `node.App`

`App` is the composition root, not a domain service. `New` normalizes and
validates configuration, prepares per-instance paths, loads the stable node ID,
and hashes the configured artifact. `Run` binds the public listener, starts the
runtime worker, constructs the coordinator and facade, publishes self
membership, starts discovery/reconciliation, and coordinates shutdown.

### `facade.Router`

`Router` decides where an HTTP request belongs:

- cluster reads and announce are node-local;
- runtime lifecycle, generation, and stage calls go to the local runtime proxy;
- coordinator APIs run locally on the leader and reverse-proxy from followers.

It does not plan deployments or execute model layers.

### `coordinator.Server`

`Server` is the HTTP adapter for coordinator-owned APIs. It decodes requests,
maps domain errors to HTTP/OpenAI responses, and delegates behavior to:

- `DeploymentController`, which owns desired deployment intent, immutable
  epochs, prepare/activate/publish/drain/unload transitions, and reconciliation;
- `GenerationController`, which pins an active deployment admission, resolves
  the route, creates unique request/session IDs, and starts stage-0 generation.

`deploymentState` is the synchronized state holder beneath
`DeploymentController`. `deploymentAdmission` increments an epoch's in-flight
count and releases it exactly once. `generationSession` couples the runtime
stream with that release.

### `clusterplan`

`clusterplan` is the policy core. It consumes model metadata plus a membership
snapshot and returns data:

- `RoutePreview` explains candidates and proposed stages;
- `DeploymentPlan` is the validated immutable identity used for execution;
- `Stage` assigns one node a contiguous `[layer_start, layer_end)` range.

Keeping this package free of HTTP and mutable cluster state makes planning
deterministic and directly testable.

### `runtimebridge`

The three proxies preserve the public-node boundary while forwarding to the
loopback runtime. The two typed clients let the coordinator call node facades:

- `HTTPDeploymentClient` performs load, activate, drain, unload, and status;
- `HTTPGenerationClient` starts an NDJSON generation stream.

### `stageexec.Executor`

`Executor` is intentionally diagnostic. It builds one Stagewire operation per
planned stage, validates response identity and payload transitions, and records
byte counts, CRCs, placement, and latency. `Generate` repeats `Execute` for
prefill and decode and closes every stage session afterward.

## C++ Runtime Components

| Directory | Class or record | Responsibility |
| --- | --- | --- |
| `worker` | `Config` | Parsed runtime identity, model path, compute backend, stage assignment, transport, activation encoding, and worker limits. |
| `api` | `HttpServer` | Bounded HTTP worker pool and route dispatch into `RuntimeAPI`. |
| `api` | `HttpResponse` | Status, content type, body, and connection behavior for one response. |
| `engine` | `RuntimeAPI` | Abstract HTTP-facing runtime contract used by `HttpServer` and tests. |
| `engine` | `RuntimeService` | Implements `RuntimeAPI`; translates JSON/binary contracts and delegates lifecycle to `ModelManager` and generation to `GenerationService`. |
| `engine` | `GenerationService` | Validates a generation request, selects the executable deployment, invokes local stages through `ModelManager`, and remote stages through `StageTransport`. |
| `engine` | `InferenceEngineFactory` | Registry from engine name to an `InferenceEngineParts` builder. |
| `engine` | `InferenceEngineParts` | A newly loaded `LayerExecutor` plus measured model residency. |
| `deployment` | `ModelManager` | Owns resident deployment epochs, active/draining state, model executors, stage workers, admission checks, and unload safety. Its storage is hidden behind `Impl`. |
| `deployment` | deployment records | `DeploymentIdentity`, `ModelResidency`, `DeploymentStatus`, and operation results are lifecycle data contracts. |
| `pipeline_parallel` | `GenerationRunner` | Runs prefill once and decode repeatedly across ordered stages, emits tokens, validates transitions, tracks bytes/calls, and closes sessions. |
| `pipeline_parallel` | `StageWorker` | Validates one stage request, decodes incoming activation data, invokes a `LayerExecutor`, encodes outgoing activation data, and builds the stage response. |
| `pipeline_parallel` | `LayerExecutor` | Engine-neutral interface for one local model stage. |
| `pipeline_parallel` | `LlamaCppStageExecutor` | Adapts partial-layer llama.cpp execution to `LayerExecutor`. |
| `pipeline_parallel` | `LlamaCppFullModelExecutor` | Adapts ordinary single-stage llama.cpp generation to `LayerExecutor`. |
| `pipeline_parallel` | `SyntheticActivationExecutor` | Deterministic CI-only executor for binary transport and stage-contract tests; it is not a model implementation. |
| `pipeline_parallel` | `StageAssignment` | Runtime-local stage index/count and layer range. |
| `adapters` | `LlamaCppModel` | Owns the patched llama.cpp model and reports loaded layer/tensor residency. |
| `adapters` | `LlamaCppStageAdapter` | Owns session-keyed partial-layer llama contexts and their KV state; executes prefill/decode and reaps idle sessions. |
| `adapters` | `LlamaCppAdapter` | Owns the ordinary full-model llama context used by single-stage execution. |
| `activation` | `ActivationCodec` | Strategy interface between engine-native F32 activations and wire encoding. |
| `activation` | `F32ActivationCodec` | Pass-through encoding with type validation. |
| `activation` | `F16ActivationCodec` | Converts F32 activations to F16 for transport and restores F32 before execution. |
| `activation` | `ActivationCodecFactory` | Registry from activation encoding name to codec builder. |
| `transport` | `StageTransport` | Strategy interface for one remote stage invocation. |
| `transport` | `HTTPStageTransport` | Persistent ordered HTTP/1.1 connection implementation of `StageTransport`. |
| `transport` | `StageTransportFactory` | Registry from transport name to transport builder. |
| `protocol` | protocol records | `GenerationRequest`, `GenerationStage`, `StageRequest`, and `StageResponse` define runtime wire semantics. |
| `inference` | inference records | `StageInput`, `StageOutput`, `Payload`, tensor descriptors, positions, ranges, and execution results are engine-neutral values. |

Private `Impl` classes hide sockets, threads, llama.cpp pointers, and mutable
storage from stable headers. They are implementation details rather than
additional architecture layers.

## Extension Points

The runtime uses three narrow strategy boundaries:

```text
engine name              -> InferenceEngineFactory -> LayerExecutor
stage transport name     -> StageTransportFactory  -> StageTransport
activation encoding name -> ActivationCodecFactory -> ActivationCodec
```

A new engine should not change `GenerationRunner`. A new transport should not
change model execution. A new activation representation should not change
planning beyond compatibility metadata.

## DRY And KISS Rules

- Share policy and behavior, not merely similarly shaped data. Go and C++ keep
  separate wire records because they are separate processes and languages.
- Keep HTTP handlers as translators; put lifecycle behavior in controllers and
  execution behavior in runtime services.
- Keep `node.App` as wiring. Do not add pass-through methods that only rename a
  constructor call.
- Keep factories only at real variation points: engine, transport, and
  activation encoding.
- Do not create a shared `utils` or `constants` package for trivial helpers.
- Keep the synthetic executor constrained to CI and contract validation.
- Split a method when it mixes ownership boundaries, not simply because it has
  reached an arbitrary line count.
