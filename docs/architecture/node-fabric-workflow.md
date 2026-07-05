# Node Fabric Workflow

This is the source-facing sequence for the current JetsonFabric node fabric. Long-form design journals, project memory, and experiment notes belong in `SamJSui/jetsonfabric-kb`; this document only describes how the code fits together.

## Process model

```text
client
  -> jetsonfabric-node :52415
       facade.Router
       membership.Store
       discovery sources
       election selector
       coordinator.Server when this node is leader
       runtimegateway.StageProxy
  -> jetsonfabric-runtime-worker :9090
```

There is one product process: `cmd/jetsonfabric-node`. There are no user-facing `control` or `agent` processes.

## Startup sequence

1. `cmd/jetsonfabric-node/main.go` parses flags into `node.Config`.
2. `node.NormalizeConfig` trims values, resolves `NODE_ROLE=auto`, normalizes discovery modes, and derives the advertise URL when omitted.
3. `node.New` loads or creates a stable node ID with `node.LoadOrCreateNodeID`.
4. `node.New` builds the embedded leader-only coordinator router with `coordinator.NewServer`.
5. `node.New` builds `runtimegateway.StageProxy` for the node-local runtime URL.
6. `node.New` wires `facade.NewRouter` with membership, coordinator, and stage runner handlers.
7. `App.Run` starts mDNS advertising, starts the discovery loop, and serves the public node API.

## Self member sequence

Each discovery tick creates the local member record:

```text
node.App.selfMember
  -> system.Detect
       hostname, OS, arch, WSL/dev hint, capabilities, metrics
  -> membership.Member
       node id, node name, role, API URL, runtime URL, capabilities, timestamps
  -> membership.Store.Upsert
```

The local member is authoritative for its own rich fields. Lightweight mDNS records must not overwrite richer HTTP-hydrated member data.

## Discovery sequence

```text
node.discoveryLoop
  -> node.refreshMembership
       upsert self
       prune stale peers except self
       discovery.MultiSource.Discover
           StaticSource: announce to configured seed URLs
           MDNSSource: browse _jetsonfabric._tcp.local
           HydratingSource: announce to mDNS-discovered peers
       merge discovered non-self members into membership.Store
```

mDNS only bootstraps peer addresses. HTTP announce hydrates the full member record through `/v1/cluster/announce`.

## Leader selection sequence

```text
facade.Router.leader
  -> membership.Store.List
  -> visibleMembers filters stale records
  -> election.ElectLeader
       drop invalid/stale peers
       keep roles that may lead: coordinator, jetson
       rank coordinator before jetson
       apply optional leader preference within same role
       prefer older started_at
       tie-break by stable node_id
```

Election is deterministic selection, not Raft consensus. It is good enough while deployment state is still early and mostly in-memory. Before real deployment writes, the coordinator should actively probe node and runtime readiness.

## Request routing sequence

Cluster-local routes are handled on every node:

```text
GET  /healthz
GET  /v1/cluster/members
GET  /v1/cluster/leader
POST /v1/cluster/announce
POST /v1/layer-split/stage
```

Coordinator-owned routes use the elected leader:

```text
client -> any node
  -> facade.Router.handleCoordinator
       if self is leader: coordinator.ServeHTTP
       else: reverse proxy to leader.APIURL
```

The facade makes the leader location an implementation detail for callers.

## Runtime stage sequence

For the current dopey-only layer-split path:

```text
client/coordinator
  -> POST node:52415/v1/layer-split/stage
  -> facade.Router.handleStageRun
  -> runtimegateway.StageProxy
  -> runtime:9090/v1/layer-split/stage
  -> runtime response returns through node facade
```

The runtime URL remains node-local. Other nodes should call the node API, not the raw local runtime URL.

## File interaction map

```text
cmd/jetsonfabric-node/main.go
  parses CLI/env-style Makefile inputs and starts node.App

internal/node/config.go
  owns config defaults, normalization, validation, advertise URL derivation

internal/node/roles.go
  infers semantic node role from system detection

internal/node/app.go
  composes the process: membership, discovery, coordinator, facade, runtime gateway

internal/system/*
  detects OS, WSL/dev environment, device class, compute backends, metrics

internal/membership/*
  defines Member and in-memory Store with stale pruning and rich-field preservation

internal/discovery/*
  discovers peer addresses through static seeds or mDNS and hydrates peers via announce

internal/election/*
  chooses one coordinator from fresh role-eligible members

internal/facade/router.go
  exposes public node API, cluster views, leader proxying, and local stage route

internal/coordinator/*
  handles leader-only planning/routing APIs embedded inside the selected node

internal/runtimegateway/stage.go
  proxies local stage execution requests from node API to node-local runtime

runtime/
  C++ runtime process; owns real model/layer execution boundary
```

## What stays out of this source repo

Move or keep these in `SamJSui/jetsonfabric-kb` instead of this repo:

- long-form design journals;
- experiment logs and troubleshooting narratives;
- hardware purchasing notes;
- roadmap brainstorming;
- ChatGPT context packs;
- non-source architectural debates.

The source repo should keep code, tests, build/run entrypoints, configuration examples, `AGENTS.md`, the README, and short source-facing architecture docs like this one.
