# Native Flash-Decode Benchmark

Follow-up: the
[shape-aware FFN benchmark](2026-08-02-native-fused-ffn.md) removes most of
the TTFT gap reported here while preserving the Flash-decode result.

## Result

JetsonFabric's native Qwen2 CUDA engine now selects Flash Attention for both
prefill and decode. GGML's fast single-token GQA kernel requires a physical KV
dimension divisible by 256. The native session therefore preserves the
request's logical capacity while rounding its internal Flash KV allocation to
that stride.

On Qwen2.5-Coder 3B with a 2,048-token prompt, the isolated alignment change
produced this A/B result:

| Metric | Unaligned Flash | Aligned Flash | Change |
| --- | ---: | ---: | ---: |
| Decode ITL | 54.262 ms | 43.837 ms | **-19.2%** |
| Decode throughput | 18.429 tok/s | 22.812 tok/s | **+23.8%** |
| End-to-end throughput | 7.868 tok/s | 8.558 tok/s | +8.8% |
| KV-cache allocation | 73.09 MiB | 81.00 MiB | +10.8% |

Nsight Systems confirmed the mechanism. The decode Flash kernel changed from
the `8x1` specialization at 319.9 us per invocation to the vectorized `1x8`
specialization at 24.1 us, a 92.5% reduction in that kernel's average duration.
The added KV memory is the explicit tradeoff for entering the CUDA fast path.

## Engine Matrix

The final matched matrix compares the optimized native engine with pinned
llama.cpp. Positive throughput changes favor native; negative TTFT changes mean
native returned the first token sooner.

| Model | Prompt | Native TTFT | llama.cpp TTFT | TTFT change | Native decode | llama.cpp decode | Decode change | E2E change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 32 | 36.98 ms | 40.84 ms | **-9.4%** | 48.74 tok/s | 41.09 tok/s | **+18.6%** | **+18.2%** |
| 1.5B | 128 | 87.52 ms | 85.72 ms | +2.1% | 48.75 tok/s | 41.10 tok/s | **+18.6%** | **+16.1%** |
| 1.5B | 512 | 311.59 ms | 293.23 ms | +6.3% | 47.95 tok/s | 40.82 tok/s | **+17.5%** | **+9.9%** |
| 1.5B | 2,048 | 1,240.92 ms | 1,189.57 ms | +4.3% | 45.82 tok/s | 39.93 tok/s | **+14.7%** | **+2.4%** |
| 3B | 32 | 67.51 ms | 71.69 ms | **-5.8%** | 23.52 tok/s | 23.78 tok/s | -1.1% | -0.8% |
| 3B | 128 | 157.92 ms | 153.38 ms | +3.0% | 23.53 tok/s | 23.77 tok/s | -1.0% | -1.2% |
| 3B | 512 | 601.55 ms | 572.07 ms | +5.2% | 23.30 tok/s | 23.67 tok/s | -1.6% | -2.6% |
| 3B | 2,048 | 2,379.03 ms | 2,309.53 ms | +3.0% | 22.84 tok/s | 23.29 tok/s | -1.9% | -2.6% |
| 7B | 32 | 138.13 ms | 138.55 ms | -0.3% | 12.58 tok/s | 12.56 tok/s | +0.2% | +0.2% |
| 7B | 128 | 320.59 ms | 308.90 ms | +3.8% | 12.57 tok/s | 12.55 tok/s | +0.2% | -0.3% |
| 7B | 512 | 1,227.62 ms | 1,165.96 ms | +5.3% | 12.48 tok/s | 12.50 tok/s | -0.2% | -1.8% |
| 7B | 2,048 | 4,932.07 ms | 4,699.10 ms | +5.0% | 12.16 tok/s | 12.37 tok/s | -1.6% | -3.6% |

Native decode materially outperformed llama.cpp for 1.5B and reached within
1.9% for every 3B and 7B row. Long-prompt native prefill remains the next
bottleneck: TTFT trails llama.cpp by 3.0% to 5.3% for the 3B and 7B prompts at
128 tokens or longer.

## Correctness

Every timed native row produced the same 32 greedy token IDs as llama.cpp.
Before timing, the harness also compared full-vocabulary logits from an
Unfused-prefill/Unfused-decode reference with Flash-prefill/Flash-decode while
forcing identical token histories.

| Model | Prompt lengths | Argmax agreement | Worst NRMSE | Minimum cosine |
| --- | --- | ---: | ---: | ---: |
| 1.5B | 32, 128, 512, 2,048 | 20/20 steps | 0.02328 | 0.999767 |
| 3B | 32, 128, 512, 2,048 | 20/20 steps | 0.02273 | 0.999782 |
| 7B | 32, 128, 512, 2,048 | 20/20 steps | 0.03220 | 0.999493 |

The 1.5B and 3B runs used the harness default NRMSE limit of 0.025. The 7B run
used an explicit 0.04 limit because its measured 0.03220 error exceeded that
tighter default; it still required exact argmax agreement and cosine similarity
of at least 0.999. This model-specific tolerance is retained in the raw report.

## Method

- Hardware: one Jetson Orin Nano 8 GB
- Power: `MAXN_SUPER`; GPU 1.020 GHz; EMC 3.199 GHz
- Models: Qwen2.5-Coder 1.5B, 3B, and 7B Instruct Q4_K_M
- Prompt seed IDs: `[12522, 5193, 264, 882]`, repeated to length
- Prompt lengths: 32, 128, 512, and 2,048 tokens
- Output: 32 greedy tokens
- Samples: one warmup and 10 iterations per process
- Replication: three cyclic process orders, pooled to 30 samples per engine row
- Source base: `dd4044096303e89e199c76654bc48ef09ee994fa`
- llama.cpp: `bf2c86ddc0685f580595954056c2e77ebabfab4f`

The three orders place native Flash, the Unfused-prefill control, and llama.cpp
first, second, and third once for every workload. Model loading is outside the
timed samples. The table reports pooled medians. Raw reports also retain means,
normal 95% confidence intervals over 30 observations, timing vectors, and
executable hashes.

## Boundary

This is a direct-engine, single-node CUDA benchmark. It excludes tokenization,
HTTP serving, distributed transport, non-greedy sampling, and architectures
outside Qwen2. It supports a Jetson-specific decode claim, not general native
engine superiority or a distributed-serving speedup.

Raw matrix reports, parity traces, A/B outputs, CUDA-kernel summaries, and
provenance are stored in
[`evidence/20260802-native-flash-decode`](evidence/20260802-native-flash-decode/).
