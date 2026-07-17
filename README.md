# JetsonFabric

Distributed LLM inference at the edge for Jetson-class devices.

JetsonFabric is a peer-discovered fabric. Every machine runs the same Go process,
`jetsonfabric-node`. A node owns identity, discovery, membership, coordinator
election, placement planning, public API routing, and supervision of a node-local
C++ runtime worker.

The runtime worker hosts an inference-engine adapter such as `llama.cpp`. The
runtime is JetsonFabric infrastructure; llama.cpp is the initial engine.

## Current architecture

```text
OpenAI-compatible client
  -> any jetsonfabric-node
  -> elected coordinator
  -> ordered stage plan
  -> target node facade
  -> node-local jetsonfabric-runtime-worker
  -> assigned llama.cpp layer range
  -> activation sent to the next stage
  -> final sampled token
```

The main packages are:

- `internal/discovery`: static and mDNS peer discovery.
- `internal/membership`: fresh local member views.
- `internal/election`: deterministic coordinator election.
- `internal/facade`: any-node API and follower forwarding.
- `internal/coordinator`: OpenAI API, model registry, and planning handlers.
- `internal/clusterplan`: topology selection and layer-range arithmetic.
- `internal/stageexec`: prefill, decode, cleanup, and stage traces.
- `internal/stagewire`: binary inter-stage framing.
- `internal/runtimebridge`: Go node to local C++ runtime proxying.
- `runtime/`: C++ runtime worker and engine adapters.

## What works now

The branch implements real sequential pipeline-parallel inference:

```text
prefill
  prompt
    -> stage 0 assigned layers
    -> F32 activation
    -> stage 1 assigned layers
    -> sampled token 1

decode
  token 1
    -> stage 0 using its persistent context
    -> F32 activation
    -> stage 1 using its persistent context
    -> sampled token 2
```

CI uses two real node processes, two supervised C++ runtime workers, two real
llama.cpp engine instances, and a tiny GGUF model. It requires two split-stage
greedy tokens to match a one-runtime baseline.

The llama.cpp integration is pinned to commit
`bf2c86ddc0685f580595954056c2e77ebabfab4f` and carries a narrow patch that:

- executes only `[layer_start, layer_end)`;
- exports hidden activations from non-final ranges;
- imports activations on downstream ranges;
- keeps final normalization, logits, and sampling on the last stage.

Initially supported graph architectures are `llama` and `qwen2`.

## Topology

A route reports objective topology:

```json
{
  "topology": "colocated",
  "stage_count": 2,
  "logical_node_count": 2,
  "physical_host_count": 1
}
```

`colocated` means multiple logical nodes share a physical host. `distributed`
means the selected stages occupy distinct physical hosts.

For 28 layers:

```text
stage_count=2: [0:14) [14:28)
stage_count=3: [0:10) [10:19) [19:28)
stage_count=4: [0:7)  [7:14) [14:21) [21:28)
```

## Request and session identity

One public completion has:

```text
chat/request ID
  -> one user-facing API operation

session ID
  -> persistent generation state shared by every stage

stage request ID
  -> one prefill/decode operation on one stage
```

JetsonFabric generates the session ID. All stages retain it across prefill and
decode, while every operation gets a distinct request ID for tracing.

The coordinator explicitly closes every stage session when generation ends.
Each runtime also reaps a session after five idle minutes, protecting memory when
the coordinator or network disappears before cleanup completes.

## OpenAI-compatible API

Send a non-streaming chat completion to any node:

```sh
curl http://dopey.local:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4",
    "messages": [
      {"role": "user", "content": "Explain CUDA kernels."}
    ],
    "max_tokens": 64
  }' | jq
```

The follower forwards the request to the elected coordinator. The coordinator
plans the stages, drives generation, and returns an OpenAI-style
`chat.completion` response.

For two logical stages on one development host, use the optional JetsonFabric
extension:

```json
{
  "model": "qwen2.5-coder-1.5b-q4",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 16,
  "jetsonfabric": {
    "stage_count": 2,
    "allow_colocated_stages": true
  }
}
```

The extension is unnecessary for a correctly advertised two-Jetson distributed
route. Streaming, tools, multimodal message parts, and non-greedy sampling are
not implemented yet.

## Diagnostic APIs

Inspect membership:

```sh
curl http://dopey.local:<port>/v1/cluster/members | jq
```

Preview a route:

```sh
curl "http://dopey.local:<port>/v1/routes/preview?model=qwen2.5-coder-1.5b-q4&stage_count=2" | jq
```

Run the lower-level diagnostic generation endpoint:

```sh
curl http://127.0.0.1:<port>/v1/layer-split/run \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "local-run",
    "model": "qwen2.5-coder-1.5b-q4",
    "payload": "Hello from JetsonFabric",
    "max_tokens": 8,
    "stage_count": 2,
    "allow_colocated_stages": true
  }' | jq
```

Typed payload transitions are mandatory. There is no public flag that disables
the activation contract.

## Build and test

```sh
make test
make setup
make runtime
make node
```

CUDA runtime build on Jetson Orin:

```sh
make runtime-cuda RUNTIME_CUDA_ARCH=87
```

Example stage configuration:

```sh
make run-node \
  NODE_NAME=dopey-stage0 \
  MODEL_PATH=/models/qwen.gguf \
  STAGE_INDEX=0 STAGE_COUNT=2 \
  LAYER_START=0 LAYER_END=14

make run-node \
  NODE_NAME=beehive-stage1 \
  MODEL_PATH=/models/qwen.gguf \
  STAGE_INDEX=1 STAGE_COUNT=2 \
  LAYER_START=14 LAYER_END=28
```

Use command-line Make variables after the target so they override `.env.local`.

## Model scope

The fabric may contain many model deployments. Initially, one runtime worker
loads one model artifact. A node can later supervise multiple workers if memory
allows.

All stages in one pipeline must ultimately prove that they use the same engine,
model ID, model artifact hash, architecture, layer count, JetsonFabric revision,
and llama.cpp revision. That exact identity enforcement is the next planning
hardening milestone.

## Honest limitations

The current implementation partitions **transformer execution**, but every
runtime still opens the complete GGUF. Therefore:

```text
compute partitioned: yes
model-weight memory partitioned: not yet
```

Two Jetsons can soon prove real topologically distributed compute, but they do
not yet combine memory to fit a model that cannot fit on either device alone.
Selective stage-owned weight loading is a later milestone.

The implementation is also sequential: stage 0 runs, then stage 1 runs. It does
not yet overlap microbatches or concurrent sessions for throughput.

## Remaining physical P0 gates

Before claiming physical topologically distributed CUDA inference:

1. Advertise and enforce exact runtime/model/artifact identity.
2. Attest that selected runtimes were CUDA-built and actually execute on GPU.
3. Require baseline token equivalence and activation CRC continuity.
4. Record per-stage latency, transfer bytes, throughput, memory, utilization,
   power, temperature, and throttling.
5. Run the acceptance harness on two distinct physical Jetsons.

See `docs/physical-jetson-validation.md` and
`docs/llama-cpp-partial-layer.md` for the detailed contracts.
