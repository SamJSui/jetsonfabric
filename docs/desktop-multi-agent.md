# Desktop Multi-Agent Simulation

This workflow is the bridge between the POC and P0/MVP:

- Local smoke: control and agents run on the desktop.
- POC: one agent moves to a real Jetson node and serves one full-model replica.
- P0/MVP: multiple Jetsons execute real layer_split.

Desktop simulation lets JetsonFabric prove the orchestration path before the
Jetson hardware arrives:

- multiple agents can join one control plane;
- each agent can expose an OpenAI-compatible proxy;
- the control plane can see several schedulable nodes;
- the layer_split planner can generate stage assignments from model metadata;
- benchmark records are produced through the same control-plane API.

It does not prove distributed transformer execution yet. When several desktop
agents point at one local llama.cpp server, they simulate node topology and
control-plane behavior, not real layer ownership.

## Run The Simulation

Start the model backend:

```sh
sh scripts/run-local-llama.sh --background
```

Start the control plane:

```sh
sh scripts/run-control.sh --background
```

Start three desktop agents:

```sh
sh scripts/run-desktop-agents.sh --count 3
```

Inspect registered nodes:

```sh
curl -sS http://127.0.0.1:52415/v1/nodes
```

Inspect the layer_split plan:

```sh
curl -sS "http://127.0.0.1:52415/v1/layer-split/plan?model=qwen2.5-coder-1.5b-q4"
```

For a 28-layer model across three equal agents, the expected stage ranges are:

```text
desktop-agent-1: [0,10)
desktop-agent-2: [10,19)
desktop-agent-3: [19,28)
```

Run a synthetic prompt through the planned agents:

```sh
curl -sS -X POST http://127.0.0.1:52415/v1/layer-split/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/poc-local-smoke/chat-request.json
```

The synthetic path sends an opaque payload through each planned agent stage over
the configured layer-split transport. Today that payload is text so it is easy to
inspect. In the real runtime it becomes activation bytes plus tensor metadata.

## Benchmark The Desktop Path

Run a short benchmark through the control plane:

```sh
sh scripts/bench-desktop-chat.sh --count 5 --concurrency 1
```

Artifacts:

- `data/benchmarks.jsonl`: server-side records written by the control plane.
- `data/desktop-chat-benchmark.json`: client-side latency summary from the
  benchmark command.

Stop the desktop simulation agents:

```sh
sh scripts/stop-desktop-agents.sh --count 3
```

Use this to compare:

- direct llama.cpp runtime call;
- control -> one agent -> llama.cpp;
- control -> multiple registered agents with the current single_node route;
- future control -> layer_split runtime once P0/MVP exists.

## Why This Is Worth Doing

This gives the project a real orchestration surface before real cluster
hardware exists. It exercises node registration, route planning, proxying,
observability, and benchmark recording. Those are the same control-plane
behaviors needed once the agents run on Jetsons.

The performance claim remains narrow: desktop simulation measures local runtime
and orchestration overhead. It does not claim layer_split latency or throughput
gains until P0/MVP implements actual layer execution across devices.
