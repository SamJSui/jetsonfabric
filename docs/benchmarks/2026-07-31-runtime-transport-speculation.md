# Runtime Transport and Speculation

## Method

- Hardware: two Jetson Orin Nano 8 GB nodes in `MAXN_SUPER`
- Network: wired 1 GbE
- Model: Qwen2.5-Coder 1.5B Q4_K_M
- Placement: 14/14 pipeline split, CUDA, F16 activations, F16 KV cache
- Serving load: concurrency 2, two warmups, ten measured requests
- Standard workload: `examples/genai-perf-request.json`, 128 output tokens
- Repetitive workload: `examples/speculative-repetitive-request.json`, 16
  output tokens before the model stopped

The relay and direct runs used identical binaries and generation settings.
Prompt-lookup speculation used a maximum draft width of four. Every measured
request succeeded. The speculation comparison was rerun after adding a
single-token recovery pass following rejected drafts. A composed two-runtime
test also verifies exact token equality across a real rejection and rollback.

## Direct Runtime Transport

| Transport | Aggregate output tok/s | TTFT p50 | ITL p50 | E2E p50 | Remote decode overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Node-facade relay | 64.94 | 144 ms | 31 ms | 4,194 ms | 2,453 us/call |
| Direct runtime | **65.56** | **139 ms** | **30 ms** | **4,167 ms** | **1,687 us/call** |

Direct transport removed one Go facade hop and reused persistent C++ HTTP
connections. It reduced measured remote decode overhead by 31.2%, but model
execution still dominated end-to-end time. The resulting throughput gain was
0.9% in this controlled run.

## Prompt-Lookup Speculation

| Workload | Strategy | Aggregate output tok/s | E2E p50 | Draft acceptance | Target decode passes |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard prose | None | **64.19** | **3,975 ms** | n/a | 1,270 |
| Standard prose | Prompt lookup | 48.79 | 5,269 ms | 9.5% | 1,230 |
| Repetitive pattern | None | 43.49 | 721 ms | n/a | 160 |
| Repetitive pattern | Prompt lookup | **48.02** | **618 ms** | 68.8% | 50 |

Prompt lookup is beneficial only when recent output reliably repeats prompt or
generation history. On the repetitive fixture it improved throughput by 10.4%,
reduced median latency by 14.3%, and reduced target decode passes by 68.8%. On
ordinary prose, rejected drafts caused distributed rollback and recovery,
reducing throughput by 24.0% and increasing median latency by 32.6%.

## Decision

`http_direct_v1` is the preferred trusted-LAN data path. Prompt-lookup drafting
remains a selectable strategy, but `none` remains the default. It should not be
enabled for general traffic without a workload-level acceptance gate or a more
accurate draft model. Continuous decode batching also remains opt-in because
the physical concurrency sweep did not beat unbatched execution on these
memory-constrained nodes.
