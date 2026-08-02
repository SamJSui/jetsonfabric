# Native KV-Cache Benchmark

## Result

JetsonFabric's native Qwen2 engine beat the pinned llama.cpp oracle in a matched
single-node greedy-decode microbenchmark while producing the exact same 32 token
IDs.

| Engine | TTFT p50 | ITL p50 | Decode p50 | End-to-end p50 |
| --- | ---: | ---: | ---: | ---: |
| JetsonFabric native | 43.39 ms | **21.03 ms** | **47.56 tok/s** | **46.03 tok/s** |
| llama.cpp oracle | **42.22 ms** | 26.73 ms | 37.41 tok/s | 36.74 tok/s |

Relative to the oracle, the native path increased decode throughput by 27.1%,
increased end-to-end throughput by 25.3%, and reduced ITL by 21.4%. Native TTFT
was 2.8% slower.

These numbers describe this exact direct-engine workload. They do not establish
that the native engine is generally faster than llama.cpp, and they are not
HTTP serving or distributed-cluster measurements.

## KV-Cache Ablation

The same native executable supports the previous full-prefix policy, allowing
the KV-cache change to be measured without changing the model, prompt, output,
power state, warmup, or sample count.

| Native decode policy | TTFT p50 | ITL p50 | Decode p50 | End-to-end p50 |
| --- | ---: | ---: | ---: | ---: |
| Full-prefix recompute | 43.92 ms | 42.90 ms | 23.31 tok/s | 23.29 tok/s |
| Incremental F16 KV cache | **43.39 ms** | **21.03 ms** | **47.56 tok/s** | **46.03 tok/s** |

Incremental decode increased native decode throughput by 104.0%, increased
end-to-end throughput by 97.6%, and reduced ITL by 51.0%. This measures the
improvement over JetsonFabric's old native path. It is not a difference from
llama.cpp: the llama.cpp oracle also performs prefill once and then decodes one
token at a time with a KV cache.

The remaining native-versus-llama.cpp difference in this microbenchmark comes
from the narrower Qwen2 greedy graph and execution path. In particular, native
generation performs `argmax` on CUDA and copies one 4-byte token ID to the host;
the pinned oracle harness copied a 607,744-byte F32 logits vector for host-side
selection. The native request also reuses one decode graph and scheduler. These
choices still need isolated A/B measurements before assigning a percentage to
each optimization.

## Workload

- Hardware: one 8 GB Jetson Orin Nano (`dopey`)
- Power: `MAXN_SUPER`, CPU 1.728 GHz, GPU 1.020 GHz, EMC 3.199 GHz
- Model: Qwen2.5-Coder 1.5B Instruct Q4_K_M
- Source SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`
- Prompt: `Once upon a time`
- Prompt IDs: `[12522, 5193, 264, 882]`
- Output: 32 greedy tokens
- Samples: one warmup and ten measured iterations per engine
- JetsonFabric commit: `1240c089ff4bb9f68bfa5edf04a40fed12dad731`
- llama.cpp revision: `bf2c86ddc0685f580595954056c2e77ebabfab4f`

Both harnesses use an incremental F16 KV cache and directly return the greedy
token. The native executable links GGML/GGML-CUDA but does not link `libllama`;
the separate oracle executable uses llama.cpp.

## Optimization

The native engine now owns an architecture-neutral inference-session contract.
Its Qwen2 implementation:

- caches K and V tensors in F16 between prefill and decode;
- reuses one decode graph and backend scheduler during a request;
- performs greedy `argmax` on the device and copies back one token ID;
- keeps the quantized token embedding on the host because Orin's CUDA backend
  cannot execute its `GET_ROWS` operation; and
- keeps the remaining model tensors on CUDA.

Nsight Systems exposed the largest prior bottleneck: placing the unsupported
Q4_K token embedding on CUDA caused a 131,272,704-byte device-to-host transfer
for every generated token. Host-resident embedding lookup removed that transfer
without moving the transformer layers off CUDA.

## Correctness

The benchmark fails on any oracle-token mismatch. Native CPU and CUDA runs both
matched llama.cpp. The cache is initialized before first use, and attention
reads depend explicitly on cache writes so CPU and CUDA schedulers observe the
same ordering.

Raw benchmark output and provenance are stored in
[`evidence/20260801-native-kv-cache`](evidence/20260801-native-kv-cache).
The evidence directory also contains the aggregate CUDA memcpy rows exported
from the pre-fix, post-fix, and oracle Nsight Systems SQLite reports.
