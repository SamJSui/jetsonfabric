# JetsonFabric

Distributed LLM inference at the edge for Jetson-class devices.

JetsonFabric is a peer-discovered node fabric. Every machine runs the same Go
process, `jetsonfabric-node`. A node owns identity, membership, discovery,
election, planning, API routing, and supervision of a node-local C++ runtime
worker.

The runtime worker is a JetsonFabric-owned host process. It loads an inference
engine adapter such as `llama.cpp`; future adapters may target TensorRT-LLM or
other engines. The runtime itself is not an inference engine.

## Architecture

```text
client
  -> any jetsonfabric-node
  -> elected coordinator plans ordered stages
  -> each target node forwards work to its local runtime worker
  -> runtime worker invokes its configured inference engine
```

Main packages:

- `cmd/jetsonfabric-node`: the product process run on every machine.
- `internal/discovery`: static and mDNS peer discovery.
- `internal/membership`: local fresh-member cache.
- `internal/election`: deterministic coordinator election.
- `internal/facade`: any-node API and follower proxying.
- `internal/coordinator`: model registry and route/run handlers.
- `internal/clusterplan`: count-agnostic stage planning and layer arithmetic.
- `internal/stagewire`: shared stage request/response wire contract.
- `internal/stageexec`: ordered inter-stage execution.
- `internal/runtimebridge`: node-to-runtime adaptation and proxying.
- `runtime/`: C++ runtime worker and engine adapters.

## Terminology

### Runtime and engine

```text
runtime process: jetsonfabric-runtime-worker
engine:          llama.cpp
compute backend: cpu or cuda
```

The runtime worker hosts the engine. Model profiles therefore list actual
inference engines only.

### Execution mode

Execution mode describes how inference work is distributed:

- `data_parallel`: each replica owns a complete model.
- `pipeline_parallel`: ordered transformer layer ranges are assigned to stages.
- `tensor_parallel`: tensor operations are partitioned across devices or nodes.

### Stage arithmetic

Stage position is represented only by `stage_index` and `stage_count`.

```text
is_first = stage_index == 0
is_last  = stage_index == stage_count - 1
```

For `stage_count=1`, stage index `0` is both first and last. There is no special
stage-role string.

Layer ranges are assigned for any positive stage count. For 28 layers:

```text
stage_count=2: [0:14] [14:28]
stage_count=3: [0:10] [10:19] [19:28]
stage_count=4: [0:7]  [7:14] [14:21] [21:28]
```

### Topology

Route previews report objective topology and counts:

```json
{
  "topology": "colocated",
  "stage_count": 2,
  "logical_node_count": 2,
  "physical_host_count": 1
}
```

`topology` is either `colocated` or `distributed`. Counts carry the precise
information; there is no special one-node placement category.

### Payload kinds

The shared stage wire contract defines semantic payload kinds:

- `text`: UTF-8 user-facing or compatibility text.
- `tokens`: tokenizer token IDs.
- `activation`: hidden-state tensors between transformer partitions.
- `sampled_token`: the selected next token.

Logits and KV cache remain internal to the inference engine. They are not
inter-stage wire payload kinds. Tensor metadata uses `dtype` and `shape`; text
uses `encoding=utf-8` and is not a tensor dtype.

## Current execution milestone

The coordinator and runtime path is fully exercised with two real nodes, two
real C++ runtime workers, two real `llama.cpp` CPU engine instances, and a real
GGUF model in CI.

The current inter-stage payload is still `text`:

```text
input text
  -> stage 0 full-model generation
  -> text handoff
  -> stage 1 full-model generation
  -> final text
```

The API reports this directly:

```json
{
  "inter_stage_payload_kind": "text",
  "plan": {
    "mode": "pipeline_parallel",
    "topology": "colocated",
    "stage_count": 2
  },
  "result": {
    "payload_kind": "text"
  }
}
```

This validates membership, planning, runtime supervision, node routing, and
handoff. It is not yet partial transformer execution.

The next runtime milestone is to change the inter-stage payload to
`activation` and execute only each stage's assigned layer range:

```text
text
  -> tokenizer
  -> tokens
  -> stage 0 assigned layers
  -> activation
  -> stage 1 assigned layers
  -> logits (engine-internal)
  -> sampler
  -> sampled token / decoded text
```

Each runtime keeps the KV cache for its assigned layer range local across decode
steps.

## Build and test

```sh
make test
make setup
make runtime
make node
```

CUDA runtime build on Jetson:

```sh
make runtime-cuda RUNTIME_CUDA_ARCH=87
```

## Run nodes

Example two-stage configuration:

```sh
make run-node \
  NODE_NAME=dopey-stage0 \
  STAGE_INDEX=0 \
  STAGE_COUNT=2 \
  LAYER_START=0 \
  LAYER_END=14

make run-node \
  NODE_NAME=beehive-stage1 \
  STAGE_INDEX=1 \
  STAGE_COUNT=2 \
  LAYER_START=14 \
  LAYER_END=28
```

Use command-line Make variables after the target so they override values in
`.env.local`.

Inspect membership:

```sh
curl http://dopey.local:<port>/v1/cluster/members | jq
```

Preview two stages:

```sh
curl "http://dopey.local:<port>/v1/routes/preview?model=qwen2.5-coder-1.5b-q4&stage_count=2" | jq
```

For colocated local development:

```sh
curl "http://127.0.0.1:<port>/v1/routes/preview?model=qwen2.5-coder-1.5b-q4&stage_count=2&allow_colocated_stages=true" | jq
```

Run the current end-to-end stage path:

```sh
curl -X POST http://127.0.0.1:<port>/v1/layer-split/run \
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

A direct stage request uses count arithmetic and explicit payload semantics:

```json
{
  "session_id": "session-1",
  "request_id": "request-1",
  "model_id": "qwen2.5-coder-1.5b-q4",
  "stage_index": 0,
  "stage_count": 1,
  "node_name": "dopey",
  "layer_start": 0,
  "layer_end": 28,
  "decode_step": 0,
  "payload_kind": "text",
  "encoding": "utf-8",
  "payload": "hello",
  "bytes_in": 5,
  "transport": "http_json",
  "max_tokens": 8
}
```

## CI

GitHub Actions runs:

1. `go test ./...`
2. A two-node real CPU E2E test using a cached tiny GGUF model.

The E2E validates:

- two real node processes;
- two real supervised runtime workers;
- the hosted `llama.cpp` engine on CPU;
- membership and advertised assignments;
- count-agnostic route metadata;
- both stage HTTP responses;
- typed text payload handoff;
- nonempty final generation.

Authoritative CUDA, JetPack, ARM64, and physical-network validation belongs on
self-hosted Jetson runners.
