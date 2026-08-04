# Native Distributed Model Scaling

Date: 2026-08-03

Node commit: `455ab56b5453b123cc9af77c6c641270a8de18f2`

Runtime source commit: `4faf708db918ce23fb165adfa9196f26d394fd49`

## Result

The JetsonFabric-native Qwen2 engine served Qwen2.5-Coder 3B, 7B, and 14B
across two physical Jetson Orin Nano 8 GB nodes. Every run used CUDA,
`MAXN_SUPER`, full-duplex 1 GbE, direct runtime transport, F32 activations,
F16 KV cache, two parallel sessions, and the same 128-token streaming
workload.

| Model | Native C1 | llama.cpp C1 | Change | Native C2 | llama.cpp C2 | Change | Native C2 TTFT | ITL | E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3B | 22.19 tok/s | 18.60 tok/s | **+19.3%** | 45.00 tok/s | 39.77 tok/s | **+13.1%** | 121 ms | 43 ms | 5.683 s |
| 7B | 12.03 tok/s | 11.54 tok/s | **+4.3%** | 24.06 tok/s | 22.60 tok/s | **+6.5%** | 222 ms | 81 ms | 10.639 s |
| 14B | 6.40 tok/s | 6.27 tok/s | **+2.1%** | 12.51 tok/s | 12.19 tok/s | **+2.6%** | 411 ms | 156 ms | 20.521 s |

All native rows completed 10/10 measured requests after two warmups. The
llama.cpp rows are the same model, placement, context, output length, hardware,
and network configuration from the
[2026-07-29 serving matrix](2026-07-29-serving-matrix.md). They are a
cross-run baseline, not a contemporaneous A/B. The stricter matched 1.5B A/B
is documented in the
[native distributed stage report](2026-08-03-native-distributed-stages.md).

At concurrency 2, the native pipeline retained 96.9% of the historical 3B
two-replica throughput and 97.7% of the historical 7B two-replica throughput.
Unlike replicas, the pipeline stores one contiguous model partition per node.
The 14B model cannot fit on one 8 GB node, so its 12.51 tok/s row is a capacity
result with no one-node or replica equivalent.

## Residency And Correctness

| Model | Placement | Stage 0 bytes | Stage 1 bytes | Total weight bytes |
| --- | --- | ---: | ---: | ---: |
| 3B | 21/15 | 1,139,644,416 | 959,332,352 | **2,098,976,768** |
| 7B | 16/12 | 2,530,570,240 | 2,146,549,760 | **4,677,120,000** |
| 14B | 26/22 | 4,700,495,872 | 4,281,647,104 | **8,982,142,976** |

For every model, resident stage bytes summed exactly to the package's total
weight bytes. Both stages reported `partitioned=true` and `pinned=true`.
Correctness requests generated token IDs `[11, 1052, 572, 264]` and text
`", there was a"`. Each activation handoff had matching byte counts and CRC32
values on the sender and receiver.

The controlled 3B-to-7B live switch also passed. The replacement became epoch
2 before epoch 1 was retired. This validates the coordinator fix that probes
direct runtime health and compatibility instead of comparing a newly prepared
deployment with the runtime's still-active predecessor.

## Method And Provenance

- Models: Qwen2.5-Coder 3B, 7B, and 14B Instruct Q4_K_M
- Source SHA-256:
  `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7`,
  `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`,
  and `c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940`
- Context: 1,536 tokens
- Workload: identical chat prompt, greedy sampling, 128 requested output tokens
- Samples: two warmups and ten measured requests per model and concurrency
- Node binary SHA-256 on both hosts:
  `c84b5f13ff83575fbb76d5a1a6d2757b523f883d9174283a73df6629c3cd49d4`
- Runtime binary SHA-256 on both hosts:
  `eab6fe95d051b1d5667cd6890bb9490de624c5d90956a873123823b61b40eba7`

Models were cold-started between size suites so 8 GB nodes did not need to
hold old and replacement partitions simultaneously. This preserves the
default zero-downtime deployment behavior while avoiding replacement-overlap
memory as a benchmark confounder.

## Raw Evidence

The [evidence directory](evidence/20260803-native-scaling/) contains C1, C2,
and correctness JSON for each model. The serving files retain every request's
latency, TTFT, ITL, stage timings, byte counts, request hash, timestamps, and
success status.
