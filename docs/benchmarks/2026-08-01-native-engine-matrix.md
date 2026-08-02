# Native Engine Workload Matrix

## Result

JetsonFabric's experimental native Qwen2 engine matched the pinned llama.cpp
oracle's greedy token IDs in all 12 tested workloads on one Jetson Orin Nano.
Across four prompt lengths and three output lengths, native decode throughput
was 9.1% to 26.4% higher, with a 20.5% median increase.

| Prompt tokens | Output tokens | Native TTFT | llama.cpp TTFT | Native decode | llama.cpp decode | Native E2E change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 42.75 ms | 43.73 ms | 47.25 tok/s | 37.38 tok/s | +25.0% |
| 32 | 128 | 42.53 ms | 43.70 ms | 46.67 tok/s | 37.57 tok/s | +23.9% |
| 32 | 512 | 42.52 ms | 43.75 ms | 43.27 tok/s | 35.64 tok/s | +21.3% |
| 128 | 32 | 94.95 ms | 93.11 ms | 46.06 tok/s | 37.34 tok/s | +20.2% |
| 128 | 128 | 94.92 ms | 93.37 ms | 45.47 tok/s | 37.53 tok/s | +20.4% |
| 128 | 512 | 94.79 ms | 93.26 ms | 42.98 tok/s | 35.33 tok/s | +21.5% |
| 512 | 32 | 365.00 ms | 336.50 ms | 42.63 tok/s | 35.61 tok/s | +10.5% |
| 512 | 128 | 365.37 ms | 336.76 ms | 42.88 tok/s | 35.80 tok/s | +16.7% |
| 512 | 512 | 365.21 ms | 336.77 ms | 40.51 tok/s | 34.94 tok/s | +15.3% |
| 2,048 | 32 | 1,994.59 ms | 1,637.36 ms | 34.54 tok/s | 29.97 tok/s | -7.6% |
| 2,048 | 128 | 2,003.38 ms | 1,637.71 ms | 34.05 tok/s | 30.12 tok/s | +2.1% |
| 2,048 | 512 | 2,006.82 ms | 1,638.53 ms | 32.34 tok/s | 29.63 tok/s | +6.0% |

Values are per-cell medians. Every decode-throughput mean has a non-overlapping
normal 95% confidence interval relative to the corresponding llama.cpp mean.
The raw report retains all timing samples and confidence intervals.

## Interpretation

Native decode won every tested cell, but native prefill did not. TTFT was 2.2%
to 2.8% lower for 32-token prompts, then became 1.6% to 2.0% higher at 128
tokens, 8.4% to 8.5% higher at 512 tokens, and 21.8% to 22.5% higher at 2,048
tokens. Native end-to-end throughput was higher in 11 of 12 workloads; the
2,048-token prompt with only 32 output tokens was 7.6% slower because decode did
not run long enough to recover the prefill deficit.

This identifies the next optimization target: native prompt processing and
prefill graph construction. It does not support the broader claim that the
native engine is always faster than llama.cpp.

The native path uses a Qwen2-specific greedy graph, retains an incremental F16
KV cache, performs argmax on CUDA, and returns one token ID to the host. The
oracle uses llama.cpp's ordinary model and context APIs with greedy selection.
Both receive the same explicit token IDs, so tokenizer behavior is excluded.
The benchmark proves output equivalence for these sequences, not general model
quality.

## Workload

- Hardware: one 8 GB Jetson Orin Nano (`dopey`)
- Power: `MAXN_SUPER`, GPU 1.020 GHz, EMC 3.199 GHz
- Model: Qwen2.5-Coder 1.5B Instruct Q4_K_M
- Source SHA-256: `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`
- Prompt seed IDs: `[12522, 5193, 264, 882]`, repeated to each prompt length
- Prompt lengths: 32, 128, 512, and 2,048 tokens
- Output lengths: 32, 128, and 512 greedy tokens
- Samples: one warmup and 20 measured iterations per engine and workload
- JetsonFabric revision: `09bfd72`
- llama.cpp revision: `bf2c86ddc0685f580595954056c2e77ebabfab4f`

Engine order alternated by workload. The runner rejected model-hash, prompt-ID,
or generated-token mismatches and checkpointed the report after every cell.
The run processed 342,720 prompt tokens and generated 112,896 output tokens
across both engines.

## Upstream Control

The pinned upstream `llama-bench` executable ran its standard prompt-processing
and token-generation tests on the same model and CUDA build. Each test used one
warmup and five measured repetitions.

| llama-bench test | Average throughput | Standard deviation |
| --- | ---: | ---: |
| pp32 | 778.20 tok/s | 99.28 tok/s |
| pp128 | 1,421.50 tok/s | 56.94 tok/s |
| pp512 | 1,533.13 tok/s | 19.06 tok/s |
| pp2048 | 1,252.30 tok/s | 0.66 tok/s |
| tg32 | 38.17 tok/s | 0.11 tok/s |
| tg128 | 38.23 tok/s | 0.02 tok/s |
| tg512 | 36.30 tok/s | 0.04 tok/s |

`llama-bench` excludes tokenization and sampling. Its zero-depth generation
rates were 1.7% to 2.1% above the matched oracle's 32-prompt-token rates, so
oracle harness overhead does not explain the matrix's 19.2% to 23.8% short-
prompt native decode advantage relative to this upstream control. The control
is not used for exact pairwise claims because its synthetic token stream and
zero-depth generation setup differ from the matched matrix.

This is a direct-engine CUDA benchmark, not HTTP serving, distributed execution,
HumanEval, or an official MLPerf result. HumanEval becomes meaningful for the
native path after tokenizer, detokenizer, and serving lifecycle integration;
before that point it would use llama.cpp to provide enough of the path that the
result would not isolate the native engine.

Raw benchmark output and executable hashes are stored in
[`evidence/20260801-native-engine-matrix`](evidence/20260801-native-engine-matrix).
