# Native Flash-Prefill Benchmark

## Result

JetsonFabric's native Qwen2 engine now uses a phase-specific attention strategy:
fused GGML Flash Attention for CUDA prefill and the established Unfused path for
single-token decode. On one Jetson Orin Nano, Flash reduced 2,048-token median
TTFT by 37.4% and prefill scratch memory by 20.8% relative to the same native
engine using Unfused attention.

| Prompt tokens | Flash TTFT | Unfused TTFT | Flash change | llama.cpp TTFT | Flash vs llama.cpp | Flash scratch change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 36.96 ms | 37.86 ms | -2.4% | 41.11 ms | -10.1% | +7.1% |
| 128 | 87.35 ms | 90.06 ms | -3.0% | 86.09 ms | +1.5% | +6.6% |
| 512 | 312.05 ms | 359.74 ms | -13.3% | 293.62 ms | +6.3% | +4.8% |
| 2,048 | 1,239.98 ms | 1,982.20 ms | **-37.4%** | 1,189.51 ms | +4.2% | **-20.8%** |

At 2,048 prompt tokens, fused prefill reduced end-to-end latency enough to
increase native end-to-end throughput by 34.6% over Unfused. It reduced the
previous native prefill deficit substantially but did not beat llama.cpp at
that length.

Decode intentionally remained Unfused, so Flash did not materially change
native decode throughput. Compared with llama.cpp, native decode was 16.9%,
14.0%, and 3.7% faster for 32-, 128-, and 512-token prompts, then 13.0% slower
at 2,048 tokens. Long-context decode remains the next engine bottleneck.

The preceding exact-shape graph-reuse change reduced TTFT by 12.3%, 4.8%,
1.3%, and 0.6% at the same four prompt lengths. At 2,048 tokens, 1,977.5 ms of
the resulting 1,979.6 ms TTFT was compute, which established fused attention as
the next target. That A/B evidence is retained in
[`evidence/20260801-native-prefill`](evidence/20260801-native-prefill/).

## Correctness

The benchmark rejects a Flash result unless GGML assigns every Flash Attention
operator to the selected CUDA backend. All 240 measured Flash generations
passed that proof.

Before timing each prompt length, a separate gate compared full-vocabulary
logits from Unfused-prefill/Unfused-decode and
Flash-prefill/Unfused-decode. Both paths consumed identical forced tokens, so
the comparison covered prefill, KV-cache writes, and four decode steps.

| Prompt tokens | Argmax agreement | Worst normalized RMSE | Minimum cosine similarity |
| ---: | ---: | ---: | ---: |
| 32 | 5/5 steps | 0.02328 | 0.999767 |
| 128 | 5/5 steps | 0.01423 | 0.999904 |
| 512 | 5/5 steps | 0.01357 | 0.999927 |
| 2,048 | 5/5 steps | 0.01602 | 0.999912 |

All timed Native Flash, Native Unfused, and llama.cpp runs also produced the
same 32 greedy token IDs for each prompt.

The physical run used acceptance thresholds of NRMSE at most 0.05 and cosine
similarity at least 0.999. Because the measured worst NRMSE was 0.02328, the
harness default was tightened after the run to 0.025; the retained observations
also pass that calibrated bound.

## Method

- Hardware: one 8 GB Jetson Orin Nano (`dopey`)
- Power: `MAXN_SUPER`; GPU 1.020 GHz; EMC 3.199 GHz
- Model: Qwen2.5-Coder 1.5B Instruct Q4_K_M
- Model SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`
- Prompt seed IDs: `[12522, 5193, 264, 882]`, repeated to length
- Prompt lengths: 32, 128, 512, and 2,048 tokens
- Output: 32 greedy tokens
- Samples: one warmup and 20 iterations per process
- Replication: three cyclic process orders per workload, pooled to 60 samples per arm
- Native revision: `9f01fe2`
- llama.cpp revision: `bf2c86ddc0685f580595954056c2e77ebabfab4f`

The three process orders place each engine first, second, and third once for
every workload. This controls per-row launch-order, temperature, and power-state
effects instead of rotating order only between different prompt lengths.

`tegrastats` recorded 2,842 samples at 500 ms. Peak reported RAM was 2,732 MiB;
swap peaked at 28 MiB from a 26 MiB baseline; peak GPU utilization was 100%; peak GPU
temperature was 65.7 C; and input power averaged 14.3 W with a 19.9 W peak.

## Boundary

This is a direct-engine CUDA benchmark for one Qwen2-family model. It excludes
tokenization, detokenization, HTTP serving, distributed stage transport,
sampling beyond greedy argmax, and other model architectures. It demonstrates
a measured Jetson-specific prefill improvement, not general superiority over
llama.cpp.

Raw timings, executable hashes, parity traces, and telemetry are stored in
[`evidence/20260802-native-flash`](evidence/20260802-native-flash/).
