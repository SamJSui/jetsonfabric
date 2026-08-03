# Native Distributed Stage Benchmark

Date: 2026-08-03

Tested commit: `4faf708db918ce23fb165adfa9196f26d394fd49`

## Result

The JetsonFabric-native Qwen2 engine now executes partial model stages on two
physical Jetsons. For Qwen2.5-Coder 1.5B Q4_K_M, dopey held layers `[0, 18)`
and grumpy held layers `[18, 28)`. The two resident partitions summed exactly
to the package's 1,111,370,240 weight bytes and 339 tensors.

In a matched native-versus-llama.cpp serving A/B, native improved aggregate
output throughput by 15.1% at concurrency 1 and 15.6% at concurrency 2. It also
reduced median TTFT, ITL, and end-to-end latency. Both engines generated the
same expected greedy tokens.

| Engine | Concurrency | Success | Output tok/s | TTFT p50 | ITL p50 | E2E p50 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp | 1 | 10/10 | 36.68 | 94 ms | 26 ms | 3.488 s | 3.491 s |
| native | 1 | 10/10 | **42.21** | **76 ms** | **22 ms** | **3.031 s** | **3.035 s** |
| llama.cpp | 2 | 10/10 | 75.10 | 93 ms | 25 ms | 3.399 s | 3.440 s |
| native | 2 | 10/10 | **86.79** | **76 ms** | **22 ms** | **2.947 s** | **2.963 s** |

The C2 native pipeline is the highest aggregate throughput measured in this
repository for this hardware. It is 8.4% above the earlier 80.06 tok/s
two-replica result, although that historical row is not part of this controlled
A/B.

## What Changed

- The JFM loader maps input tensors only on stage 0, output tensors only on the
  final stage, and transformer tensors only for the assigned layer interval.
- The native Qwen2 graph accepts either token IDs or incoming F32 activations
  and returns either outgoing activations or a sampled token.
- Prefill, incremental decode, and rollback keep stage-local F16 KV state.
- The engine-neutral stage executor now admits native multi-stage assignments
  and verifies that coordinator placement matches resident JFM layers.

No coordinator request carries token-by-token activations. Stage 0 calls the
peer runtime directly through `http_direct_v1` using the existing Stagewire
contract.

## Residency And Correctness

| Node | Layers | Resident bytes | Resident tensors |
| --- | ---: | ---: | ---: |
| dopey | 0-18 | 641,912,832 | 217 |
| grumpy | 18-28 | 469,457,408 | 122 |
| Sum | 0-28 | **1,111,370,240** | **339** |

The physical validator passed for both engines with expected token IDs
`[11, 1052, 572, 264]`, generated text `", there was a"`, two CUDA-active
hosts, distributed topology, and activation CRC continuity within each run.
Native F32 activation payloads were 24,576 bytes for the four-token prefill and
6,144 bytes per decode token.

## Matched Method

- Hardware: two Jetson Orin Nano 8 GB developer kits
- Power mode: `MAXN_SUPER`
- Network: 1,000 Mb/s full-duplex wired Ethernet
- Model: Qwen2.5-Coder 1.5B Instruct Q4_K_M
- Source SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`
- Placement: 18/10 layers
- Context: 1,536 tokens
- Workload: identical chat prompt, greedy sampling, 128 requested output tokens
- Transport: `http_direct_v1`, F32 activations
- Runtime: CUDA, F16 KV cache, two parallel sessions, decode batch size 1
- Samples: two warmups and ten measured requests per engine and concurrency
- Node binary SHA-256: `71fe4d54c7e2914f452cb3180d6c2c3c5bea5c55556e9d03ee2dceaaf8aa0384`
- Runtime binary SHA-256: `eab6fe95d051b1d5667cd6890bb9490de624c5d90956a873123823b61b40eba7`

Only the engine, model identifier, and artifact representation changed between
the native JFM and llama.cpp GGUF runs. Both representations use the same
source model hash. The measured percentages describe this model and workload;
they do not establish the same gain for larger models or other architectures.

## Raw Evidence

- [`native-c1.json`](evidence/20260803-native-distributed/native-c1.json) - native C1
- [`native-c2.json`](evidence/20260803-native-distributed/native-c2.json) - native C2
- [`llama-c1.json`](evidence/20260803-native-distributed/llama-c1.json) - llama.cpp C1
- [`llama-c2.json`](evidence/20260803-native-distributed/llama-c2.json) - llama.cpp C2

The raw files retain per-request TTFT, ITL, latency, stage timing, byte counts,
request hashes, timestamps, and success or failure status.
