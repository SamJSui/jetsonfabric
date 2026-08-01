# Native Inference Correctness Baseline

## Result

JetsonFabric's native engine reproduced the exact four-token greedy sequence
from the pinned llama.cpp oracle for Qwen2.5-Coder 1.5B Q4_K_M on both CPU and
CUDA. The CUDA run used one 8 GB Jetson Orin Nano in `MAXN_SUPER` with clocks
locked.

| Backend | TTFT p50 | ITL p50 | Decode p50 | End-to-end p50 |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 100.84 ms | 115.42 ms | 8.66 tok/s | 8.94 tok/s |
| CPU | 114.05 ms | 161.41 ms | 6.20 tok/s | 6.68 tok/s |

Under this narrow workload, CUDA reduced TTFT by 11.6%, reduced ITL by 28.5%,
and increased measured decode throughput by 39.8% versus CPU. Both cases used
one warmup and ten measured iterations. These percentages compare the two
native backends; they are not comparisons against llama.cpp serving.

## Correctness Contract

- Model: Qwen2.5-Coder 1.5B Q4_K_M
- Source SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`
- Prompt: `Once upon a time`
- Prompt token IDs: `[12522, 5193, 264, 882]`
- Expected and observed greedy IDs: `[11, 1052, 572, 264]`
- Oracle: llama.cpp `bf2c86ddc0685f580595954056c2e77ebabfab4f`

The native benchmark fails if any generated token differs from the oracle. The
native binary links GGML/GGML-CUDA, but not `libllama`; llama.cpp is used only
by the separate correctness-oracle executable.

## Interpretation

This is the first real-model baseline for JetsonFabric-owned model loading,
Qwen2 graph construction, CUDA backend selection, and greedy generation. It is
not yet a serving benchmark. The engine accepts token IDs, loads the full model
on one node, and recomputes the full prefix for every generated token. It does
not yet have tokenizer, KV-cache, lifecycle, or distributed-stage integration.

Accordingly, `decode_tokens_per_second` is explicitly labeled
`full_prefix_recompute` and must not be compared with KV-cached llama.cpp decode
rates. The next meaningful performance gate is KV-cache parity; after that,
the same architecture strategy can expose stage boundaries for two-node
execution.

## Physical Evidence

The CUDA run reached 99% sampled GPU utilization, 12.536 W peak input power,
1,922 MiB peak reported RAM use, and 52.75 C peak GPU temperature. The native
load phase took 13.607 seconds with JFM hash verification and a warm Linux page
cache; no cache eviction was performed.

Raw evidence is stored in
[`evidence/20260801-native-inference`](evidence/20260801-native-inference).

