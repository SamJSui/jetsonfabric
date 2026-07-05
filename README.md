# JetsonFabric

Distributed LLM inference at the edge for Jetson-class devices, starting with
Jetson Orin Nano nodes and CUDA-backed local runtimes.

JetsonFabric is moving from an explicit `control + agent` prototype into a
**peer-discovered node fabric with an elected coordinator**. Each machine runs a
`jetsonfabric-node`; nodes discover each other with mDNS/static fallback, elect
one coordinator, expose the same API surface, and forward compute work to a
node-local runtime.

Current architecture version: **node fabric preview**.

## Goals

- Run useful LLM inference on low-cost edge hardware.
- Make multiple Jetson Orin Nano nodes feel like one observable serving fabric.
- Use network discovery with mDNS instead of passing IPs around whenever the LAN
  supports multicast.
- Keep leader/coordinator selection internal to the cluster.
- Keep CUDA/runtime execution local to each node.
- Build toward real pipeline parallelism where ordered transformer layer stages
  are assigned to nodes through deployment plans.

## Architecture

```text
client / curl / OpenAI-compatible caller
  -> any jetsonfabric-node :52415
  -> follower proxies public API to elected coordinator
  -> coordinator plans routing/deployment
  -> target node forwards stage work to local runtime
  -> local C++/CUDA runtime executes assigned work
```

Main pieces:

- `cmd/jetsonfabric-node`: the product process run on each machine.
- `internal/discovery`: static seed discovery, mDNS discovery, and hydration via
  HTTP announce.
- `internal/membership`: in-memory member table with stale-member pruning.
- `internal/election`: deterministic leader election over fresh eligible members.
- `internal/facade`: public node API, follower proxying, local stage routing.
- `internal/coordinator`: leader-only planning/control role embedded in a node.
- `internal/runtimegateway`: node-to-local-runtime stage proxy.
- `runtime/`: C++ runtime worker and pipeline-parallel stage execution shell.
- `tools/bench`: developer benchmark client, not part of the production fabric.

## Node fabric model

Every real machine runs the same command:

```sh
make node-run NODE_CLUSTER_ID=home-lab NODE_NAME=dopey
make node-run NODE_CLUSTER_ID=home-lab NODE_NAME=beehive
```

Each node:

- owns a stable node ID under its data directory;
- advertises itself over mDNS as `_jetsonfabric._tcp.local`;
- periodically discovers peers;
- exchanges full member records through `/v1/cluster/announce`;
- participates in deterministic leader election if control-eligible;
- exposes public cluster/API routes on port `52415`;
- forwards `/v1/layer-split/stage` to its local runtime.

mDNS is used only to bootstrap peer addresses. After a peer is discovered, the
node performs an HTTP announce/handshake to hydrate the full membership record
with capabilities, metrics, engine metadata, and fresh timestamps.

## Leader election

Current election is deterministic, not Raft consensus:

1. remove stale members;
2. keep only `control_eligible` nodes;
3. sort by highest `control_priority`;
4. break ties by stable `node_id`.

This is intentionally simple for the current homelab/edge prototype. Before real
deployment writes and long-running layer execution, the coordinator should also
actively probe node health and runtime readiness. Membership means "may exist";
readiness means "can receive work now."

## Pipeline parallelism direction

Pipeline parallelism requires strict layer order. Discovery does not decide that
order. The elected coordinator creates a deployment plan such as:

```text
stage 0: dopey    layers 0:14
stage 1: beehive  layers 14:28
```

The current runtime path is intentionally conservative:

- first prove one-node `pipeline_parallel` on `dopey` with `stage_count=1`;
- forward stage requests through the node facade to `127.0.0.1:9090`;
- keep the C++ runtime responsible for model/layer execution;
- later replace HTTP/JSON activation transfer with a persistent binary data
  plane for hot-path runtime-to-runtime communication.

## Quick start: dopey + beehive discovery

On `dopey`:

```sh
git pull
go test ./...

make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=dopey \
  NODE_DISCOVERY=mdns \
  NODE_DATA_DIR=.cache/jetsonfabric-dopey \
  NODE_CONTROL_ELIGIBLE=true \
  NODE_CONTROL_PRIORITY=20
```

On `beehive`:

```sh
git pull
go test ./...

make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=beehive \
  NODE_DISCOVERY=mdns \
  NODE_DATA_DIR=.cache/jetsonfabric-beehive \
  NODE_CONTROL_ELIGIBLE=false \
  NODE_CONTROL_PRIORITY=0
```

From Windows CMD, PowerShell, WSL, or another client:

```sh
curl http://dopey.local:52415/v1/cluster/members
curl http://dopey.local:52415/v1/cluster/leader
```

If `.local` or multicast is unavailable from the client environment, use a LAN or
Tailscale address for the node you are querying. WSL is a dev client by default,
not a required cluster member.

## Quick start: dopey runtime stage gateway

Start the local runtime on `dopey`:

```sh
make runtime-run \
  NODE_NAME=dopey \
  RUNTIME_LISTEN=127.0.0.1:9090 \
  RUNTIME_MODE=pipeline_parallel \
  STAGE_INDEX=0 \
  STAGE_COUNT=1 \
  LAYER_START=0 \
  LAYER_END=28
```

Start the node in another terminal:

```sh
make node-run \
  NODE_CLUSTER_ID=home-lab \
  NODE_NAME=dopey \
  NODE_DISCOVERY=mdns \
  NODE_DATA_DIR=.cache/jetsonfabric-dopey \
  NODE_CONTROL_ELIGIBLE=true \
  NODE_CONTROL_PRIORITY=20
```

Then test the node-to-runtime stage route:

```sh
curl -sS -X POST http://127.0.0.1:52415/v1/layer-split/stage \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "test-session",
    "request_id": "test-request",
    "model_id": "qwen2.5-coder-1.5b-q4",
    "stage_index": 0,
    "node_name": "dopey",
    "role": "single",
    "layer_start": 0,
    "layer_end": 28,
    "decode_step": 0,
    "shape": [1, 1, 1],
    "dtype": "synthetic",
    "payload": "hello",
    "bytes_in": 5,
    "transport": "http"
  }'
```

Until real transformer layer execution is wired, the runtime should return an
honest `layer_executor_not_implemented` style error. That still proves the node
API reaches the local runtime.

## Build and test

```sh
go test ./...
make build
```

`make build` builds the node binary and C++ runtime. The benchmark client is a
developer tool and is intentionally not part of the production build path.

```sh
make bench BENCH_URL=http://127.0.0.1:52415/v1/chat/completions
```

## Documentation

- [AGENTS.md](AGENTS.md) captures coding-agent instructions and project
  boundaries.
- [docs/architecture/node-discovery-control-plane.md](docs/architecture/node-discovery-control-plane.md)
  captures the node fabric architecture.
- [docs/architecture/mdns-discovery.md](docs/architecture/mdns-discovery.md)
  captures the mDNS discovery design.
- [docs/p0-layer-split-mvp.md](docs/p0-layer-split-mvp.md) captures the MVP for
  real layer-split inference.
- [docs/engineering-standards.md](docs/engineering-standards.md) captures the
  implementation quality bar.

## Non-goals for the current version

- Do not claim an edge cluster beats frontier cloud models.
- Do not treat mDNS membership as scheduling-grade health.
- Do not use WSL as a production cluster node.
- Do not start with full Raft consensus.
- Do not keep HTTP/JSON activation transfer as the final data plane.
