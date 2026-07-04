# Node Discovery, Leader Election, and Control Plane Architecture

This document sketches the refactor from explicit `control` + `agent` processes toward an Exo-like `jetsonfabric-node` UX.

The product goal is:

```text
Run the same node command on every machine.
Send prompts to any node.
JetsonFabric discovers peers, elects one active coordinator, and routes work through an ordered deployment plan.
```

The key point is that `node` becomes the product primitive. `control` and `agent` stop being user-facing roles and become internal responsibilities.

---

## Current architecture

Current JetsonFabric has explicit processes:

```text
jetsonfabric-control
  - loads model registry
  - owns routing/planning API
  - listens on :52415

jetsonfabric-agent
  - requires a fixed control URL
  - optionally proxies to a local runtime engine
  - heartbeats to the configured control plane

jetsonfabric-runtime-worker
  - C++ runtime process
  - local compute/stage execution boundary
```

This is easy to reason about, but the UX requires remembering which machine is the control plane and how every node reaches it.

---

## Target architecture

Each machine runs:

```text
jetsonfabric-node
```

A node process contains:

```text
jetsonfabric-node
  discovery
  membership store
  leader election
  public facade API
  coordinator role if elected leader
  local runtime gateway
  optional runtime supervisor
```

Usually, each compute node also has its own local C++ runtime process:

```text
jetsonfabric-node              public API on :52415
jetsonfabric-runtime-worker    local runtime on 127.0.0.1:9090
```

Every node is coordinator-capable, but only one node is the active coordinator/leader at a time.

```text
leader node:
  active coordinator
  planner
  deployment owner
  public OpenAI-compatible API

follower node:
  discovery and membership
  local runtime access
  public API facade
  proxies coordinator requests to the leader
```

---

## Proposed Go package layout

```text
cmd/
  jetsonfabric-node/
    main.go

  jetsonfabric-control/       legacy/manual wrapper for now
  jetsonfabric-agent/         legacy/manual wrapper for now

internal/
  node/
    app.go                    lifecycle composition
    config.go                 node config
    identity.go               stable node id
    lifecycle.go              start/stop orchestration

  membership/
    member.go                 Member type
    store.go                  in-memory member table
    health.go                 stale member pruning

  discovery/
    discovery.go              Source interface
    static.go                 seed-based discovery first
    mdns.go                   later

  election/
    election.go               deterministic election now, consensus later if needed

  facade/
    router.go                 public API on every node
    proxy.go                  follower-to-leader reverse proxy

  coordinator/
    server.go                 evolved control server role
    deployments.go            deployment lifecycle
    planner.go                route and stage planning
    routing.go                chat/deployment routing

  deployment/
    plan.go                   DeploymentPlan
    stage.go                  StageAssignment
    store.go                  deployment state

  runtimegateway/
    client.go                 local runtime client
    stage.go                  node internal stage route
    health.go                 runtime health

  runtimeprocess/
    supervisor.go             optional local runtime process manager later
```

Naming guidance:

```text
node          product process
coordinator   leader-only control role inside a node
facade        public API exposed by every node
membership    shared cluster view
runtime       local C++ compute process
```

---

## Member record

Each node advertises a member record:

```go
type Member struct {
    ClusterID string

    NodeID   string
    NodeName string

    APIURL     string
    RuntimeURL string

    ControlEligible bool
    ControlPriority int

    OS           cluster.OperatingSystem
    Arch         string
    Capabilities map[string]any
    Engines      []cluster.EngineEndpoint

    StartedAt time.Time
    LastSeen  time.Time
}
```

`NodeID` should be generated once and stored on disk, for example:

```text
/var/lib/jetsonfabric/node-id
```

Do not use hostname as identity. Hostnames are labels, not stable IDs.

---

## Discovery and networking

Discovery only answers:

```text
who exists?
how can I reach them?
what cluster do they claim to belong to?
what capabilities do they advertise?
```

Discovery does not decide pipeline order and does not create deployments.

### Phase 1: static seed discovery

Start with static seed URLs. This is deterministic, debuggable, and works over Tailscale or LAN.

```bash
jetsonfabric-node \
  --cluster-id home-lab \
  --node-name grumpy \
  --advertise-url http://100.x.x.x:52415 \
  --seeds http://100.116.162.40:52415
```

Node-to-node endpoints:

```text
POST /v1/cluster/announce
GET  /v1/cluster/members
GET  /v1/cluster/leader
```

Startup flow:

```text
grumpy starts
  -> loads stable node_id
  -> detects local capabilities
  -> registers itself locally
  -> announces to seed dopey
  -> receives dopey's known member table
  -> merges members
  -> runs election
```

### Phase 2: mDNS discovery

mDNS is LAN-local service discovery. Nodes advertise a service record, and peers subscribe to those records.

Service:

```text
_jetsonfabric._tcp.local
```

TXT fields:

```text
cluster_id=home-lab
node_id=<stable id>
node_name=dopey
api_url=http://dopey.local:52415
control_eligible=true
control_priority=10
```

mDNS differs from static seeds because there is no configured peer list. A node asks the LAN, "who advertises `_jetsonfabric._tcp.local`?" and then filters discovered peers by `cluster_id`.

mDNS is useful for:

```text
zero-config LAN clusters
Mac mini / Jetson homelab discovery
finding nodes after IP changes
```

mDNS is not enough for:

```text
cross-subnet discovery
Tailscale-only networks without multicast forwarding
WAN discovery
strong membership agreement
```

So the discovery manager should support multiple sources:

```go
type Source interface {
    Discover(ctx context.Context) ([]membership.Member, error)
}
```

Implementations:

```text
StaticSource   now
MDNSSource     later
```

---

## Leader election options

JetsonFabric should start with deterministic leader election, not Raft.

### Deterministic election

Every node runs the same pure function over its membership table:

```text
healthy control-eligible members
  sorted by:
    highest control priority
    stable node_id tie-breaker
```

Pseudo-code:

```go
func ElectLeader(now time.Time, members []membership.Member, staleAfter time.Duration) (membership.Member, bool) {
    candidates := make([]membership.Member, 0, len(members))

    for _, m := range members {
        if !m.ControlEligible {
            continue
        }
        if now.Sub(m.LastSeen) > staleAfter {
            continue
        }
        candidates = append(candidates, m)
    }

    if len(candidates) == 0 {
        return membership.Member{}, false
    }

    slices.SortFunc(candidates, func(a, b membership.Member) int {
        if a.ControlPriority != b.ControlPriority {
            return cmp.Compare(b.ControlPriority, a.ControlPriority)
        }
        return strings.Compare(a.NodeID, b.NodeID)
    })

    return candidates[0], true
}
```

Properties:

```text
simple
cheap
debuggable
no extra protocol
works well for early homelab clusters
```

Limitations:

```text
not true consensus
can split-brain during network partitions
no durable committed log
no strong deployment-state safety
```

This is acceptable while deployment state is in-memory and the system is early.

### Voting

Voting means nodes explicitly exchange votes and elect a leader by majority or quorum.

Typical flow:

```text
node detects no leader
  -> increments term/epoch
  -> asks peers for votes
  -> peers grant or reject vote
  -> candidate becomes leader if it wins enough votes
```

Voting is stronger than deterministic election because nodes record who they voted for in a term. It can reduce accidental dual leadership, but by itself it still needs careful handling of terms, timeouts, and partitions.

Voting adds:

```text
vote request/response RPCs
term/epoch tracking
candidate state
leader heartbeats
quorum rules
```

### Consensus

Consensus means the cluster agrees not just on a leader, but also on an ordered sequence of state changes.

For JetsonFabric, that would mean agreeing on events like:

```text
create deployment dep-001
assign dopey stage 0
assign grumpy stage 1
mark deployment ready
remove failed node
```

Consensus is about committed cluster state, not merely discovery.

### Raft

Raft is a consensus algorithm that combines leader election and replicated logs.

A Raft-based JetsonFabric control plane would look like:

```text
Raft nodes elect a leader
leader accepts deployment writes
leader appends writes to a replicated log
followers replicate the log
entry is committed after quorum replication
all nodes apply committed entries in the same order
```

Properties:

```text
stronger correctness
clear leader terms
safer failover
consistent deployment state
```

Costs:

```text
more code
more state management
persistent log/snapshots
quorum requirement
more operational complexity
more difficult debugging on weak edge devices
```

Recommended path:

```text
P1: deterministic election
P2: leader lease / epoch to reduce split-brain risk
P3: Raft only if deployment state needs production-grade consistency
```

---

## Facade API

Every node exposes the same public API on `:52415`.

If local node is leader:

```text
serve coordinator routes directly
```

If local node is follower:

```text
serve local health/cluster status
proxy public coordinator routes to leader
```

Examples:

```text
GET  /healthz                         local
GET  /v1/cluster/members              local or leader-backed
GET  /v1/cluster/leader               local
POST /v1/chat/completions             leader or proxy to leader
POST /v1/deployments                  leader or proxy to leader
GET  /v1/deployments                  leader or proxy to leader
```

User experience:

```bash
curl http://any-node:52415/v1/chat/completions
```

---

## Pipeline parallelism and deployment plans

Discovery does not determine pipeline order. The active coordinator creates a deployment plan.

A deployment plan is the source of truth for layer order:

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
      "next_node_name": "grumpy"
    },
    {
      "stage_index": 1,
      "role": "last",
      "node_name": "grumpy",
      "layer_start": 14,
      "layer_end": 28,
      "prev_node_name": "dopey"
    }
  ]
}
```

Pipeline parallelism requires ordered stages:

```text
stage 0 runs before stage 1
stage 1 runs before stage 2
KV cache lives with each stage's layers
final stage produces logits or sampled token
```

So the leader must distribute assignments:

```text
dopey:
  deployment_id=dep-qwen-001
  stage_index=0
  layers=0:14
  next=grumpy

grumpy:
  deployment_id=dep-qwen-001
  stage_index=1
  layers=14:28
  prev=dopey
```

---

## Control plane vs data plane

Control plane:

```text
HTTP/JSON
discovery
membership
election
deployment planning
health
dashboard
```

Data plane:

```text
runtime-to-runtime activation transfer
persistent binary transport eventually
hidden-state tensors
KV-cache-bearing decode loop
```

Do not keep JSON activation payloads in the hot path forever. The runtime server is fine as a process boundary, but pipeline activations should eventually move over an optimized binary data plane.

---

## Recommended implementation order

1. Add `cmd/jetsonfabric-node` with single-node self-leader mode.
2. Add `membership.Store` and stable node identity.
3. Add deterministic `election.ElectLeader`.
4. Add `facade.Router` that serves directly when leader and proxies when follower.
5. Add static seed discovery through `/v1/cluster/announce` and `/v1/cluster/members`.
6. Add deployment plan data model and deployment APIs.
7. Add node-to-local-runtime configuration path.
8. Add mDNS discovery.
9. Add leader lease or Raft only if consistency requirements demand it.
10. Add optimized runtime-to-runtime binary data plane for pipeline activations.

---

## UX target

Old:

```bash
make control-run
make runtime-run
make agent-run CONTROL_URL=http://control:52415 ...
```

New:

```bash
jetsonfabric-node
```

Then:

```bash
curl http://any-node:52415/v1/chat/completions
```

The node that receives the request either handles it as leader or proxies it to the leader.
