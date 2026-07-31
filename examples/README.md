# Request Examples

These files are executable request bodies for the public node API. Run the
commands from the repository root and replace the node URL when necessary.

Activate a registered model as a two-stage deployment:

```sh
curl -sS -X POST http://dopey.local:52415/v1/deployments/switch \
  -H 'Content-Type: application/json' \
  --data-binary @examples/deployment-switch-request.json | jq
```

Send a buffered chat completion through any cluster node:

```sh
curl -sS -X POST http://grumpy.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json | jq
```

The chat API currently supports `model`, `messages`, `max_tokens`,
`max_completion_tokens`, `stream`, and the optional `jetsonfabric` routing
object. Unsupported OpenAI fields are not included in the tracked example.

`genai-perf-request.json` is the fixed 128-token chat workload used by the
concurrent serving benchmark. It is a reproducibility fixture, not a separate
API contract.

Run the checked-in benchmark suites against the active deployment:

```sh
make bench \
  BENCH_MANIFEST=examples/benchmark-manifest.json \
  BENCH_OUTPUT=data/benchmarks/local-serving-smoke.json
```

The manifest fixes endpoint assignment, request bytes, warmup count,
concurrency, timeout, and streaming behavior. For an independent-replica
baseline, give a suite both node endpoints; measured requests are assigned
round-robin:

```json
"endpoints": [
  "http://node-a.local:52415/v1/chat/completions",
  "http://node-b.local:52415/v1/chat/completions"
]
```

Both nodes must be prepared as independent single-node deployments for that
baseline. Two endpoints in the manifest do not change cluster topology.

Each measured response includes aggregated runtime timing by phase and stage:

- `execution` is model-engine work;
- `activation_decode` and `activation_encode` are codec work;
- `stage_total` includes stage validation, conversion, codec, and execution;
- `remote_call` is the caller-observed remote stage round trip;
- `remote_overhead` is `remote_call - stage_total`.

`remote_overhead` intentionally combines connection-queue wait, request/frame
serialization, HTTP handling, and network transfer. It must not be presented
as pure wire latency.

Compute quality-adjusted EvalPlus goodput from a normalized evaluation report:

```sh
python3 tools/bench/evalplus_goodput.py \
  --input data/benchmarks/humaneval-20260727/humaneval-summary.json \
  --output data/benchmarks/humaneval-20260727/evalplus-goodput.json
```

The output reports passed tasks per wall-clock minute. It does not infer
correctness from serving throughput; the pass counts must come from EvalPlus.
