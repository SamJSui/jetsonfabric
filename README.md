# JetsonFabric

Distributed pipeline-parallel LLM inference for NVIDIA Jetson clusters.

JetsonFabric makes a group of Jetson devices behave like one logical inference
system. Every machine runs the same Go node process for discovery, membership,
coordinator election, deployment planning, and the public API. Each node
supervises a C++ runtime worker that executes its assigned transformer layers
through `llama.cpp` and CUDA.

Two physical Jetson Orin Nano 8 GB nodes now run one model across contiguous
layer ranges over wired Ethernet. Qwen2.5-Coder 32B Q2_K demonstrates an
11.46 GiB resident-weight capacity that cannot fit on either node. A balanced
1.5B pipeline also demonstrates a concurrent serving benefit, while one node
remains faster for a single active request.

JetsonFabric is an experimental source release, not production-ready software.
Traffic currently assumes a trusted LAN and installation is source-based.
Detailed contracts and operator guides are indexed in
[docs/README.md](docs/README.md).

## How It Works

```text
OpenAI-compatible client
  -> any jetsonfabric-node
  -> elected coordinator
  -> deployment plan
  -> stage-0 runtime GenerationRunner
  -> local assigned layers
  -> binary Stagewire activation
  -> peer runtime directly, or peer node facade in relay mode
  -> final logits and sampled token
  -> buffered response or SSE stream
```

For a two-stage pipeline:

```text
prefill: prompt -> stage 0 -> activation -> stage 1 -> sampled token
decode:  token  -> stage 0 -> activation -> stage 1 -> sampled token
```

The coordinator handles control-plane policy once per request. Runtime workers
own the latency-sensitive prefill/decode loop, direct activation transfer,
stage-local KV state, sampling, cancellation, and cleanup.

Implemented behavior includes:

- identical `jetsonfabric-node` processes with static and mDNS discovery;
- deterministic coordinator election and fresh peer membership;
- versioned deployment plans with model hashes, epochs, and layer assignments;
- partial-layer `llama.cpp` execution for Llama and Qwen2-family models;
- stage-local model-weight residency and persistent prefill/decode contexts;
- registered F32 and F16 activation codecs over binary Stagewire transport,
  with byte-count and CRC validation;
- dynamic load, activate, drain, and unload without restarting the node;
- active-deployment repair after an empty runtime process restarts;
- bounded runtime HTTP workers and persistent ordered connections to peer nodes;
- selectable relay and direct runtime-to-runtime HTTP transports;
- opt-in prompt-lookup speculative decoding with target verification and
  distributed KV-cache rollback;
- OpenAI-compatible buffered and streaming chat completions through any node.

See [the architecture](docs/architecture.md) for component ownership and
[Stagewire v2](docs/stagewire-v2.md) for the inter-stage frame contract.

## Measured Results

The physical test cluster used two Jetson Orin Nano 8 GB nodes in MAXN_SUPER
mode over wired 1 GbE. All models were Qwen2.5-Coder Q4_K_M. The performance
sweep used a 1,024-token context, 64 output tokens, three warmups, and 20
measured requests at concurrency 1.

![One-node and two-node throughput and energy across model sizes](docs/benchmarks/figures/model-scaling.svg)

HumanEval used one greedy sample for each of 164 EvalPlus tasks, a 1,536-token
context, and at most 512 output tokens. The 1.5B, 3B, and 7B models ran on one
node; 14B used the two-node pipeline.

![HumanEval Plus quality versus fixed-output throughput](docs/benchmarks/figures/quality-throughput.svg)

| Model | Placement | TTFT p50 | ITL p50 | E2E p50 | Output tok/s | J/token | HumanEval+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | One node | 103 ms | 24 ms | 1.68 s | 37.51 | 0.452 | 68.3% |
| 3B | One node | 126 ms | 41 ms | 2.81 s | 22.43 | 0.838 | 76.8% |
| 7B | One node | 248 ms | 79 ms | 5.31 s | 12.05 | 1.718 | 84.1% |
| 14B | Two nodes | 418 ms | 159 ms | 10.52 s | 6.08 | 4.479 | 86.6% |

For this original sweep, TTFT ended at the first nonempty SSE content chunk and
ITL measured intervals between streamed content chunks. The current benchmark
client instead tracks runtime token events and verifies their count against
final usage. E2E is full request latency for the fixed 64-token experiment. The
table and performance figure report medians; tail estimates are not promoted
from a 20-request sample.

Key findings:

- 32B Q2_K is the measured dense-model capacity limit: 11.46 GiB of resident
  weights load as 33/31 layer partitions and sustain all 164 HumanEval
  generations at up to concurrency two without a CUDA OOM.
- 7B is the measured performance/quality knee. It delivered about twice the
  14B throughput with 38% of its energy per output token.
- The 7B-to-14B HumanEval difference was not statistically distinguishable on
  164 tasks.
- Splitting reduced throughput by 20.5% for 1.5B, 14.1% for 3B, and 8.8% for
  7B relative to each model's one-node baseline.
- Exact greedy tokens, activation CRC continuity, CUDA activity, partitioned
  residency, power, temperature, and failure behavior were captured physically.

These measurements establish the current claim boundary: aggregate model
capacity is proven. The original concurrency-1 sweep does not establish a
distributed speedup.

### Capacity Frontier

The maximum-model experiment compared Qwen2.5-Coder 32B Q2_K against the 14B
Q4_K_M quality/performance point:

| Model | Placement | HumanEval | HumanEval+ | Fixed-64 output tok/s | J/token |
| --- | --- | ---: | ---: | ---: | ---: |
| 14B Q4_K_M | Two nodes | 89.6% | 86.6% | 6.08 | 4.48 |
| 32B Q2_K | Two nodes | **90.9%** | 86.6% | 2.39 | 9.50 |

The 32B model proves additional capacity, but not a better Pareto point:
HumanEval+ tied, the base-score difference was not statistically significant,
and fixed-output throughput fell 60.7%. At concurrency two, 32B reached 5.149
aggregate output tok/s while both GPUs remained active. See the
[32B capacity report](docs/benchmarks/2026-07-30-32b-capacity.md).

### Serving Matrix

A follow-up matrix used F16 activations, a 1,536-token context, 128 output
tokens, stage allocations balanced from measured runtime timings, and
concurrency 1, 2, and 4. Values below are the best measured concurrency-2
configurations after warmup.

| Model | Placement | Output tok/s | TTFT p50 | ITL p50 | E2E p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.5B | Two replicas | 80.06 | 75 ms | 24 ms | 3.19 s |
| 1.5B | 18/10 pipeline | 63.44 | 148 ms | 29 ms | 4.02 s |
| 3B | Two replicas | 46.45 | 115 ms | 42 ms | 5.48 s |
| 3B | 21/15 pipeline | 39.77 | 203 ms | 48 ms | 6.44 s |
| 7B | Two replicas | 24.63 | 198 ms | 80 ms | 10.36 s |
| 7B | 16/12 pipeline | 22.60 | 316 ms | 85 ms | 11.33 s |
| 14B | 26/22 pipeline | **12.19** | 533 ms | 159 ms | 20.99 s |

Replicas remain the throughput optimum when the model fits on one board. The
balanced pipeline retains 79.2%, 85.6%, and 91.8% of replica throughput for
1.5B, 3B, and 7B respectively. For 14B, which cannot fit on one 8 GB node,
concurrency 2 nearly doubles aggregate pipeline throughput over concurrency 1
while median request latency rises only 3.0%.

The detailed method, every suite, runtime timing decomposition, and claim
boundaries are in the
[serving matrix report](docs/benchmarks/2026-07-29-serving-matrix.md).

Direct runtime transport and prompt-lookup speculation are evaluated in the
[runtime transport report](docs/benchmarks/2026-07-31-runtime-transport-speculation.md).

## Quick Start

### Requirements

- Linux development host or NVIDIA Jetson;
- Go toolchain specified by `go.mod`;
- CMake and a C++20 compiler;
- `curl`, `jq`, and standard build utilities;
- a compatible GGUF model;
- CUDA toolkit for GPU builds on Jetson.

### Run One Node

```sh
git clone https://github.com/SamJSui/jetsonfabric.git
cd jetsonfabric
cp .env.example .env.local
```

Set the model identity and local artifact path:

```dotenv
MODEL=qwen2.5-coder-1.5b-q4
MODEL_PATH=/absolute/path/to/model.gguf
```

For CPU-only development:

```dotenv
RUNTIME_COMPUTE_BACKEND=cpu
RUNTIME_N_GPU_LAYERS=0
```

Build, test, and launch one node supervising one full-model runtime:

```sh
make setup
make test
make dev-up
```

Successful startup ends with:

```text
JetsonFabric development node is ready.

Model:    qwen2.5-coder-1.5b-q4
Layers:   [0,28)
Pipeline: stage 0 of 1
Backend:  cpu
Node:     http://127.0.0.1:19180
Runtime:  http://127.0.0.1:19190
```

Inspect the node and issue a completion from another terminal:

```sh
make dev-status
make dev-chat DEV_PROMPT='Explain JetsonFabric in one sentence.' DEV_MAX_TOKENS=16
make kill
```

`make dev-status` includes health, membership, runtime state, and a route
preview. `make dev-chat` prints an OpenAI-compatible response:

```json
{
  "object": "chat.completion",
  "model": "qwen2.5-coder-1.5b-q4",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "JetsonFabric ..."
      },
      "finish_reason": "length"
    }
  ]
}
```

Generated IDs, timestamps, text, and token counts vary by run.

### Join Two Jetsons

Build and install the CUDA runtime on each Jetson. Both nodes need the same
model artifact hash and cluster token; the local artifact path may differ.

```sh
make node-linux-arm64
make runtime-cuda RUNTIME_CUDA_ARCH=87
sh scripts/install-node-layout.sh
```

Start both nodes with `--runtime-idle`. Once membership converges, switch a
two-stage deployment and prompt through either node:

```sh
curl -sS http://dopey.local:52415/v1/cluster/members | jq

curl -sS -X POST http://dopey.local:52415/v1/deployments/switch \
  -H 'Content-Type: application/json' \
  --data-binary @examples/deployment-switch-request.json | jq

curl -sS -X POST http://grumpy.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json | jq
```

Deployment requests may optionally provide measured layer counts, for example
`"stage_layer_counts": [18, 10]`. Counts must be positive, match the stage
count, and sum to the model's transformer layer count. Equal layer allocation
remains the default.

`dopey` and `grumpy` are example hostnames. Requests sent to a non-coordinator
node are forwarded through the node facade. The [node join guide](docs/node-join.md)
contains the complete trusted-LAN configuration and model-registry example.

## Development

The primary build and validation commands are:

```sh
gofmt -w ./cmd ./internal ./tools
go test ./...
make build
```

Explicit targets:

```sh
make node
make runtime
make runtime-cuda RUNTIME_CUDA_ARCH=87
```

Real-model integration:

```sh
make test-integration-single \
  MODEL_PATH=/absolute/path/to/model.gguf \
  MODEL=qwen2.5-coder-1.5b-q4

make test-integration-pipeline \
  MODEL_PATH=/absolute/path/to/model.gguf \
  MODEL=qwen2.5-coder-1.5b-q4
```

Pipeline activations use lossless `f32` by default. Select the registered `f16`
codec to halve activation bytes at each inter-stage boundary:

```sh
make test-integration-pipeline \
  MODEL_PATH=/absolute/path/to/model.gguf \
  MODEL=qwen2.5-coder-1.5b-q4 \
  RUNTIME_ACTIVATION_ENCODING=f16
```

All nodes in a deployment must advertise the same activation encoding. Engines
continue to consume and produce `f32`; the runtime codec owns only the wire
conversion, independently of the engine and stage transport.

Configuration stays separated by purpose:

| Path | Purpose |
| --- | --- |
| `.env.example` | Developer-machine environment template. |
| `configs/models.example.json` | Model registry and planning metadata example. |
| `examples/*.json` | Executable API and benchmark request bodies. |

Do not commit GGUF files, cluster tokens, host-specific paths, generated plans,
or mutable node state.

## Roadmap

The distributed runtime, physical CUDA proof, sustained 32B capacity run,
runtime restart repair, and persistent peer connection baseline are complete.
Current work is:

1. **Harden the memory boundary:** estimate weights, KV, activations, compute
   buffers, fragmentation, and replacement overlap before admitting a plan.
2. **Improve two-node utilization:** profile decode kernels and evaluate
   pipeline micro-batching or continuous batching against throughput, latency,
   and energy.
3. **Evaluate a sparse frontier:** extend partial residency to one measured MoE
   architecture only if its accuracy and active-parameter speed justify the
   added runtime graph support.
4. **Operator and recovery experience:** package services, distribute models,
   expose repeatable benchmarks, and automate node-loss acceptance tests.

After those are stable:

- add per-node identity, TLS, and secure cluster admission;
- persist coordinator intent and deployment state across leader failure;
- add structured metrics and traces;
- evaluate additional measured activation codecs;
- pursue tensor parallelism only if measurements and Jetson networking justify
  the added complexity.

## Current Limitations

- Partial-layer support is limited to Llama and Qwen2-family graphs through a
  patch tied to the pinned `llama.cpp` revision.
- Reported model residency covers tensor payloads, not allocator overhead,
  compute buffers, KV cache, fragmentation, or replacement overlap.
- Runtime HTTP serving is bounded to two workers by default. Model execution
  remains serialized inside each stage adapter.
- Peer operations reuse one ordered HTTP/1.1 connection per target; they are
  not multiplexed and do not overlap microbatches.
- The `f16` activation codec uses scalar CPU conversion; physical benchmarks
  must determine whether lower wire volume offsets conversion cost.
- Chat completions use greedy sampling.
- Stage traffic uses a shared cluster token over plaintext HTTP. Use only a
  trusted network.
