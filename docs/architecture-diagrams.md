# Architecture Diagrams

These diagrams describe the intended JetsonFabric shape for P0 and the later
layer-split runtime. The P0 path uses an agent proxy to a node-local
`llama-server`. The layer-split diagrams are a design target, not a claim that
real distributed layer execution exists yet.

## Component View

```mermaid
flowchart LR
  client[OpenAI-compatible client]
  control[jetsonfabric-control]
  registry[Model registry]
  benchmarks[Benchmark recorder]

  subgraph dopey[Jetson dopey]
    agentDopey[jetsonfabric-agent]
    llamaDopey[llama-server]
    runtimeDopey[future jetsonfabric-runtime]
  end

  subgraph grumpy[Jetson grumpy]
    agentGrumpy[jetsonfabric-agent]
    runtimeGrumpy[future jetsonfabric-runtime]
  end

  client -->|POST /v1/chat/completions| control
  control --> registry
  control --> benchmarks
  control -->|P0 route to advertised backend| agentDopey
  agentDopey -->|local proxy| llamaDopey

  agentDopey -->|heartbeat node_name=dopey| control
  agentGrumpy -->|heartbeat node_name=grumpy| control

  control -. P1 plan .-> agentDopey
  control -. P1 plan .-> agentGrumpy
  agentDopey -. launch or monitor .-> runtimeDopey
  agentGrumpy -. launch or monitor .-> runtimeGrumpy
  runtimeDopey -. activation tensors .-> runtimeGrumpy
```

## Go Contract View

This is a struct-level view of the main Go contracts. It is not meant to mirror
every field; it shows ownership and dependency direction.

```mermaid
classDiagram
  class ControlServer {
    +Router()
    +handleHeartbeat()
    +handleChatCompletions()
    +handleLayerSplitPlan()
  }

  class AgentClient {
    +SendHeartbeat()
  }

  class AgentServer {
    +Router()
    +handleChatCompletions()
    +handleLayerSplitStage()
  }

  class NodeRecord {
    +node_name
    +hostname
    +capabilities
    +metrics
    +backends
    +last_seen
  }

  class HeartbeatRequest {
    +node_name
    +hostname
    +capabilities
    +metrics
    +backends
  }

  class RuntimeBackend {
    +id
    +kind
    +base_url
    +models
    +openai_compatible
  }

  class ModelProfile {
    +id
    +runtime
    +layer_count
    +min_memory_gb
    +placement_modes
  }

  class RouteMetadata {
    +mode
    +node_name
    +model_id
    +backend_id
    +stages
  }

  class BenchmarkRecord {
    +model_id
    +node_name
    +route_mode
    +latency_ms
    +output_tokens
  }

  class RuntimeClient {
    +CompleteChat()
  }

  ControlServer --> NodeRecord
  ControlServer --> ModelProfile
  ControlServer --> RouteMetadata
  ControlServer --> BenchmarkRecord
  NodeRecord --> RuntimeBackend
  AgentClient --> HeartbeatRequest
  AgentServer --> RuntimeClient
```

## Agent Join And Heartbeat

```mermaid
sequenceDiagram
  autonumber
  participant Agent as jetsonfabric-agent
  participant System as Jetson OS
  participant Runtime as node-local runtime
  participant Control as jetsonfabric-control

  Agent->>System: detect hostname, hardware, OS, metrics
  Agent->>Agent: node_name = --node-name or hostname
  opt llama runtime configured
    Agent->>Runtime: health/model readiness check
    Runtime-->>Agent: ready
    Agent->>Agent: advertise proxy backend
  end
  Agent->>Control: POST /v1/agent/heartbeat
  Note over Agent,Control: Authorization: Bearer join token
  Control->>Control: validate join token
  Control->>Control: upsert node by node_name
  Control-->>Agent: registered node record
```

## P0 Prompt Path

```mermaid
sequenceDiagram
  autonumber
  participant Client as Client
  participant Control as jetsonfabric-control
  participant Registry as Model registry
  participant Agent as agent on dopey
  participant Llama as llama-server on dopey
  participant Bench as Benchmark recorder

  Client->>Control: POST /v1/chat/completions
  Control->>Registry: load requested model profile
  Control->>Control: select compatible online node
  Control->>Agent: POST /v1/chat/completions
  Agent->>Llama: proxy OpenAI-compatible request
  Llama-->>Agent: model response
  Agent-->>Control: model response
  Control->>Control: attach route metadata
  Control->>Bench: record latency, route, tokens, metrics
  Control-->>Client: response with JetsonFabric route metadata
```

## Future Layer-Split Path

In the future layer-split path, the control plane plans and observes. It should
not relay activation tensors.

```mermaid
sequenceDiagram
  autonumber
  participant Client as Client
  participant Control as jetsonfabric-control
  participant AgentA as agent dopey
  participant RuntimeA as runtime dopey
  participant RuntimeB as runtime grumpy
  participant Bench as Benchmark recorder

  Client->>Control: POST /v1/chat/completions
  Control->>Control: plan layer_split route
  Control->>AgentA: start route on first stage
  AgentA->>RuntimeA: RunPrefillStage tokens, layers 0..N
  RuntimeA->>RuntimeB: activation tensor bytes + metadata
  RuntimeB->>RuntimeB: run layers N+1..final and lm_head
  RuntimeB-->>RuntimeA: logits/token result
  RuntimeA-->>AgentA: completed response
  AgentA-->>Control: final response + stage metadata
  Control->>Bench: record latency, bytes, stages, thermal data
  Control-->>Client: response with layer_split route metadata
```

## Node Name Conflict Policy

```mermaid
stateDiagram-v2
  [*] --> NewHeartbeat
  NewHeartbeat --> MissingName: node_name empty
  MissingName --> Rejected
  NewHeartbeat --> NewNode: node_name not seen
  NewNode --> Registered
  NewHeartbeat --> ExistingNode: node_name seen
  ExistingNode --> Registered: same live agent endpoint
  ExistingNode --> Conflict: different live endpoint
  ExistingNode --> Registered: old record expired
  Conflict --> Rejected
```

For P0, `node_name` is the identity. It defaults to the Jetson hostname, so lab
nodes can be named `dopey`, `grumpy`, and `sleepy`. Duplicate live names are
configuration conflicts rather than names the control plane silently rewrites.

