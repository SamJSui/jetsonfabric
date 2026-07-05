# Node Discovery, Leader Election, and Coordinator Architecture

JetsonFabric now uses a peer-discovered node fabric rather than separate user-facing coordinator and worker binaries.

The product goal is:

```text
Run the same node command on every machine.
Send prompts to any node.
JetsonFabric discovers peers, selects one active coordinator, and routes work through deployment plans.
```

`jetsonfabric-node` is the product primitive. The coordinator is an internal role held by the elected leader. Runtime execution remains behind each node through the local C++ runtime worker.

## Runtime shape

```text
client / curl / OpenAI-compatible caller
  -> any jetsonfabric-node :52415
  -> follower proxies coordinator routes to selected leader
  -> coordinator plans routing and deployments
  -> target node forwards stage work to local runtime
  -> jetsonfabric-runtime-worker executes local work
```

Each node owns:

```text
discovery
membership store
role-gated leader selection
public facade API
coordinator role when selected leader
local runtime gateway
```

Usually, a compute node also runs:

```text
jetsonfabric-node              public API on :52415
jetsonfabric-runtime-worker    local runtime on 127.0.0.1:9090
```

## Package layout

```text
cmd/
  jetsonfabric-node/          product process

internal/
  node/                       lifecycle composition and config
  membership/                 member records and in-memory membership table
  discovery/                  static seed discovery, mDNS, HTTP hydration
  election/                   deterministic role-gated leader selection
  facade/                     public API on every node
  coordinator/                leader-only planning and routing role
  runtimegateway/             node-to-local-runtime proxy
  system/                     local system and device capability detection

runtime/                      C++ runtime worker and pipeline-stage shell
tools/bench/                  developer benchmark client
```

There should not be separate `control` or `agent` product folders. Historical logic belongs under `coordinator`, `facade`, `membership`, `discovery`, or `runtimegateway` based on responsibility.

## Member record

Each node advertises one member record:

```go
type Member struct {
    ClusterID string
    NodeID    string
    NodeName  string
    Hostname  string
    Role      NodeRole

    APIURL     string
    RuntimeURL string

    LeaderPreference int

    OS           cluster.OperatingSystem
    Arch         string
    Capabilities map[string]any
    Metrics      map[string]any
    Engines      []cluster.EngineEndpoint

    StartedAt time.Time
    LastSeen  time.Time
}
```

`NodeID` is generated once and stored on disk. Hostnames are labels, not identity.

## Discovery

Discovery answers:

```text
who exists?
how can I reach them?
what cluster do they claim to belong to?
what role and capabilities do they advertise?
```

Discovery does not decide pipeline order and does not create deployments.

### Static discovery

Static discovery announces this node to configured seed URLs:

```bash
make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=grumpy \
  NODE_SEEDS=http://dopey.local:52415
```

Node-to-node endpoint:

```text
POST /v1/cluster/announce
```

The response returns the seed node's current cluster view. The caller merges the returned members into its local membership table.

### mDNS discovery

mDNS is LAN-local service discovery.

Service:

```text
_jetsonfabric._tcp.local
```

TXT fields are lightweight bootstrap metadata:

```text
cluster_id=home-lab
node_id=<stable id>
node_name=dopey
role=jetson
api_port=52415
leader_preference=0
```

After discovering a peer over mDNS, the node calls `/v1/cluster/announce` to hydrate the full member record with capabilities, metrics, engines, and fresh timestamps.

## Role-gated leader selection

Leader selection is deterministic, not Raft consensus. Every node runs the same pure function over its local membership table.

Current ranking:

```text
1. remove stale or invalid members
2. keep only leader-eligible roles: coordinator and jetson
3. prefer coordinator over jetson
4. use leader_preference only within the same role
5. prefer older started_at for stability
6. break ties by stable node_id
```

Roles:

```text
auto         derive from local system facts
jetson       can coordinate and compute
coordinator  dedicated coordinator node
worker       member/compute node, not leader-eligible
test         development/test node, not leader-eligible
```

Auto role detection:

```text
WSL/dev environment -> test
Jetson device class -> jetson
generic Linux PC -> worker
```

This removes normal reliance on numeric coordinator priority. `leader_preference` remains as an explicit advanced tie-break within the same semantic role.

## Facade API

Every node exposes the same public API on `:52415`.

If local node is leader:

```text
serve coordinator routes directly
```

If local node is follower:

```text
serve local health and cluster status
proxy public coordinator routes to the selected leader
```

Examples:

```text
GET  /healthz                         local
GET  /v1/cluster/members              local
GET  /v1/cluster/leader               local
POST /v1/cluster/announce             local
POST /v1/layer-split/stage            local runtime gateway
POST /v1/chat/completions             leader or proxy to leader
```

## Pipeline parallelism and deployment plans

Discovery does not determine pipeline order. The active coordinator creates deployment plans.

A deployment plan is the source of truth for ordered layer stages:

```json
{
  "deployment_id": "dep-qwen-001",
  "model_id": "qwen2.5-coder-1.5b-q4",
  "execution_mode": "pipeline_parallel",
  "stages": [
    {
      "stage_index": 0,
      "role": "first",
      "node_name": "dopey",
      "layer_start": 0,
      "layer_end": 14,
      "next_node_name": "beehive"
    },
    {
      "stage_index": 1,
      "role": "last",
      "node_name": "beehive",
      "layer_start": 14,
      "layer_end": 28,
      "prev_node_name": "dopey"
    }
  ]
}
```

The near-term runtime milestone is intentionally conservative: prove one-node `pipeline_parallel` on dopey with `stage_count=1`, then expand to multi-node stages after runtime execution is real.

## Future hardening

Before long-running deployments, the leader should actively probe candidates:

```text
membership freshness
node /healthz
runtime health
stage prepare ack
stage commit ack
```

After deployment state becomes durable or safety-critical, add a leader lease/epoch. Move to Raft only if JetsonFabric needs strongly consistent committed deployment state across coordinator failover.
