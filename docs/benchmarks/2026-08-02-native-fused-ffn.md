# Native Shape-Aware FFN Benchmark

## Result

The native Qwen2 CUDA engine previously represented the gated feed-forward
activation as separate SiLU and multiply graph operations. Nsight Systems
showed that this was the largest native-only prefill cost remaining after the
Flash Attention work.

JetsonFabric now selects between two GGML graph forms from the gate tensor's
element count:

- at least 10,000 elements: `ggml_swiglu_split`, which becomes one fused CUDA
  gated-activation kernel;
- fewer than 10,000 elements: separate SiLU and multiply operations, which are
  faster for the 1.5B model's small single-token decode vector.

This is a tensor-shape policy, not a model-ID allowlist. It selects fused
prefill for every tested workload, separate decode for 1.5B, and fused decode
for 3B and 7B.

## Isolated A/B

The pre-change and fused runs used Qwen2.5-Coder 3B, a 2,048-token prompt, one
output token, one warmup, and ten timed iterations.

| Metric | Separate SiLU/multiply | Fused SwiGLU | Change |
| --- | ---: | ---: | ---: |
| Median TTFT | 2,375.192 ms | 2,307.048 ms | **-2.9%** |
| Median prefill compute | 2,373.957 ms | 2,305.787 ms | **-2.9%** |
| Prefill scratch | 257.0 MiB | 343.0 MiB | +33.5% |

Nsight attributed 204.4 ms to the old SiLU and multiply kernels. The fused
gated kernel took 128.4 ms, a 37.2% reduction in that operation. The 76.0 ms
kernel reduction is consistent with the 68.2 ms end-to-end TTFT reduction.
Persistent weights and KV-cache sizes did not change.

An unconditional fused policy exposed a smaller-shape regression: 1.5B decode
fell from the prior 48.7 tok/s range to 41.4 tok/s at a 32-token prompt. The
shape-aware policy restored 48.8 tok/s while retaining the fused-prefill TTFT.

## Final Matrix

Positive throughput changes favor native. Negative TTFT changes mean native
returned the first token sooner than pinned llama.cpp.

| Model | Prompt | Native TTFT | llama.cpp TTFT | TTFT change | Native decode | llama.cpp decode | Decode change | E2E change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 32 | 36.50 ms | 40.78 ms | **-10.5%** | 48.73 tok/s | 41.12 tok/s | **+18.5%** | **+18.2%** |
| 1.5B | 128 | 84.78 ms | 85.81 ms | **-1.2%** | 48.77 tok/s | 41.11 tok/s | **+18.7%** | **+16.6%** |
| 1.5B | 512 | 299.50 ms | 293.29 ms | +2.1% | 47.98 tok/s | 40.85 tok/s | **+17.5%** | **+11.3%** |
| 1.5B | 2,048 | 1,196.82 ms | 1,189.58 ms | +0.6% | 45.85 tok/s | 39.94 tok/s | **+14.8%** | **+4.9%** |
| 3B | 32 | 66.58 ms | 71.65 ms | **-7.1%** | 24.06 tok/s | 23.78 tok/s | +1.2% | +1.5% |
| 3B | 128 | 153.23 ms | 153.34 ms | **-0.1%** | 24.06 tok/s | 23.79 tok/s | +1.2% | +1.0% |
| 3B | 512 | 582.14 ms | 571.73 ms | +1.8% | 23.77 tok/s | 23.67 tok/s | +0.4% | -0.3% |
| 3B | 2,048 | 2,313.91 ms | 2,311.01 ms | +0.1% | 23.14 tok/s | 23.29 tok/s | -0.6% | -0.3% |
| 7B | 32 | 136.69 ms | 138.56 ms | **-1.3%** | 12.67 tok/s | 12.56 tok/s | +0.9% | +0.9% |
| 7B | 128 | 313.82 ms | 308.51 ms | +1.7% | 12.67 tok/s | 12.56 tok/s | +0.9% | +0.6% |
| 7B | 512 | 1,201.98 ms | 1,165.79 ms | +3.1% | 12.57 tok/s | 12.51 tok/s | +0.5% | -0.6% |
| 7B | 2,048 | 4,848.24 ms | 4,700.11 ms | +3.2% | 12.25 tok/s | 12.36 tok/s | -1.0% | -2.3% |

Relative to the prior native build, the final policy reduced TTFT by 1.0% to
3.9%. It preserved 1.5B decode performance and improved 3B and 7B decode by up
to 2.3% and 0.7%, respectively. End-to-end throughput improved by as much as
2.4% within each model's prior matrix.

## Correctness

All 12 timed rows produced the same 32 greedy token IDs as llama.cpp. The
Flash-versus-unfused attention gate also retained exact argmax agreement over
five logits steps at every prompt length.

| Model | Parity cases | Worst NRMSE | Minimum cosine |
| --- | ---: | ---: | ---: |
| 1.5B | 4 | 0.023279 | 0.999767 |
| 3B | 4 | 0.022727 | 0.999782 |
| 7B | 4 | 0.032200 | 0.999493 |

The 1.5B and 3B runs used the default 0.025 NRMSE limit. The 7B run retained
the previously established 0.04 limit and still required exact argmax and at
least 0.999 cosine similarity.

## Method

- Hardware: one Jetson Orin Nano 8 GB
- Power: `MAXN_SUPER`; clocks locked during the run
- Models: Qwen2.5-Coder 1.5B, 3B, and 7B Instruct Q4_K_M
- Prompt seed IDs: `[12522, 5193, 264, 882]`, repeated to length
- Prompt lengths: 32, 128, 512, and 2,048 tokens
- Output: 32 greedy tokens
- Samples: one warmup and 10 iterations per process
- Replication: three cyclic engine orders, pooled to 30 samples per row
- Source base: `37a79fa1202958fdf977c9972ca6dd6e8ea491e5`
- llama.cpp: `bf2c86ddc0685f580595954056c2e77ebabfab4f`

The 3B matrix used the unconditional fused build measured immediately before
the final shape threshold. Every 3B graph exceeds that threshold, so the final
policy selects the identical graph. Final-policy 1.5B and 7B matrices used the
same native executable hash.

## Boundary

This is a direct-engine, single-node CUDA benchmark. It excludes tokenization,
HTTP serving, distributed transport, non-greedy sampling, and architectures
outside Qwen2. The result supports a Jetson-specific native-engine claim, not a
distributed-serving speedup.

Raw matrices, targeted A/B outputs, parity traces, CUDA-kernel summaries, and
provenance are stored in
[`evidence/20260802-native-fused-ffn`](evidence/20260802-native-fused-ffn/).
