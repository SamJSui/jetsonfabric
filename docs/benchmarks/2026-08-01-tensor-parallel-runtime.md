# Two-Orin Nano Tensor-Parallel Runtime

## Question

Can two Jetson Orin Nano 8 GB devices execute one Qwen2.5-Coder 14B model more
effectively with tensor sharding than with JetsonFabric's existing layer
pipeline on their current Gigabit Ethernet link?

## Method

- Hardware: two Jetson Orin Nano 8 GB developer kits in `MAXN_SUPER` with
  `jetson_clocks` enabled.
- Network: dedicated wired 1 GbE paths through the same router.
- Model: Qwen2.5-Coder 14B Instruct, GGUF Q4_K_M, 48 transformer layers.
- Model SHA-256: `c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940`.
- Source base: JetsonFabric `3626ce7` plus this tensor-runtime working-tree
  change; llama.cpp `bf2c86ddc0685f580595954056c2e77ebabfab4f`.
- Request: `Once upon a time`, 2,048-token context, 32 greedy output tokens.
- Samples: one warmup followed by three measured sequential requests per row.
- Correctness: every measured request produced the same 32-token sequence in
  pipeline and tensor modes.
- Client: `tools/bench/runtime_jsonl.py`, reading newline-delimited runtime
  token events and measuring TTFT, ITL, end-to-end latency, and decode rate.
- Telemetry: `tegrastats` at 250 ms plus Ethernet byte counters around each
  measured window.

Tensor mode used llama.cpp's quantized meta-device split. One driver runtime
owned tokenization, KV state, and sampling; a provider process exposed the
second Jetson's CUDA device through persistent GGML RPC. Pipeline mode loaded
layers 0-23 on the first Jetson and 24-47 on the second and sent F32 boundary
activations through JetsonFabric's direct runtime transport.

## Results

| Mode | Placement | E2E mean | TTFT mean | ITL mean | Decode tok/s | Output tok/s | Dominant wire bytes | J/output token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pipeline | 24/24 layers | **5.14 s** | **361 ms** | **154 ms** | **6.50** | **6.22** | **3.2 MB** | **4.25** |
| Tensor | 50/50 weights | 19.99 s | 1,086 ms | 609 ms | 1.64 | 1.60 | 6.55 GB | 11.51 |
| Tensor | 60/40 weights | 19.58 s | 1,096 ms | 596 ms | 1.68 | 1.63 | 6.55 GB | 11.36 |

Wire counters covered the complete client invocation: one warmup followed by
three measured requests. Latency, throughput, and power calculations exclude
the warmup.

The 60/40 tensor split improved decode rate only 2.3% over the 50/50 split.
The matched pipeline remained 3.9 times faster in decode, 3.0 times faster to
first token, 3.8 times faster end to end, and about 2.7 times more energy
efficient per output token.

During the 60/40 tensor window, the driver transmitted 6.55 GB and the provider
received 6.55 GB. Mean GPU utilization was only 14.8% on the driver and 10.0%
on the provider, despite both model shards remaining resident with negligible
swap. The link, not CUDA compute or model capacity, was the dominant bound.

## Finding

Real tensor sharding is correct and usable as a capacity mechanism, but it is
not on the Pareto frontier for this two-node Gigabit Ethernet cluster. The
pipeline moves one hidden-state boundary per token; llama.cpp tensor sharding
moves intermediate collective data throughout the model. At 14B, that
difference outweighed any layer-level compute parallelism.

The practical next step is to keep the pipeline as the supported two-Jetson
path and test tensor execution again only with a materially faster interconnect.
Unequal tensor splits cannot recover performance while the dominant collective
traffic continues to saturate 1 GbE.

## Implementation Boundary

This milestone adds a real direct-runtime `tensor_parallel` mode, an
engine-neutral `Executor` contract, a validated tensor device mesh, a
llama.cpp-specific tensor adapter, and a guarded CUDA RPC provider. It does not
add coordinator tensor placement or claim production security.

GGML RPC has no authentication or encryption. The provider rejects a
non-loopback listener unless `--allow-remote` is present, and the mode must only
be used on a trusted private network.

The summary used for the table is checked in beside this report as
[`2026-08-01-tensor-parallel-runtime-summary.json`](2026-08-01-tensor-parallel-runtime-summary.json).
The complete client reports and request templates are under
[`evidence/20260801-tensor-runtime`](evidence/20260801-tensor-runtime).

The measured client invocation for each row was:

```bash
JETSONFABRIC_CLUSTER_TOKEN="$CLUSTER_TOKEN" \
python3 tools/bench/runtime_jsonl.py \
  --url "http://$DRIVER:52521/v1/generate" \
  --request "$REQUEST" \
  --count 3 \
  --warmup 1 \
  --timeout 600 \
  --expected-tokens "$PIPELINE_REPORT" \
  --name "$RUN_NAME" \
  --output "$OUTPUT"
```

For the pipeline reference, `--expected-tokens` was omitted because it created
the reference sequence. Tensor runs used that pipeline report as the required
cross-mode token oracle.
