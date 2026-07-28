# Two-Orin Nano Layer-Split Benchmark

Date: 2026-07-27

## Result

JetsonFabric executed real CUDA inference across two physical 8 GB Jetson Orin
Nano nodes over wired 1 GbE. The benchmark proves:

- exact one-node/two-node token equivalence for Qwen2.5-Coder 1.5B Q4_K_M;
- layer-partitioned tensor residency, activation transfer, and CRC continuity;
- a Qwen2.5-Coder 14B Q4_0 model that cannot load on one node but does load and
  generate when split across two nodes;
- a controlled Q4_K_M scaling sweep from 1.5B through 14B;
- HumanEval and HumanEval+ pass@1 for 164 tasks at each model size;
- stable warmed latency distributions with no thermal throttling;
- a current recovery/reconciliation defect after a stage node restarts.

It does not prove a speedup. For the 1.5B model, distribution is slower and
less energy-efficient than one-node inference. Its demonstrated value is
aggregate model capacity. The 7B one-node configuration is the measured
performance/quality knee for this model family and benchmark; the 14B split
does not show a statistically distinguishable HumanEval improvement over 7B.

## Test Setup

| Item | Configuration |
| --- | --- |
| Nodes | Dopey and Grumpy, Jetson Orin Nano 8 GB |
| Network | Wired 1 GbE, Wi-Fi disabled |
| Power mode | MAXN_SUPER, `jetson_clocks` active |
| OS | Jetson Linux R39.2.0 |
| JetsonFabric | `39b1073` |
| llama.cpp | `bf2c86ddc0685f580595954056c2e77ebabfab4f` |
| Backend | CUDA, all assigned layers offloaded |
| Main model | Qwen2.5-Coder 1.5B Q4_K_M |
| Requests | 3 warmups, 30 measured, concurrency 1 |
| Output | Fixed 64 tokens per request |
| Telemetry | `tegrastats` at 100 ms plus wired interface counters |

The prompts contained 103, 2,920, and 11,368 characters. TTFT is time to the
first nonempty SSE content chunk. ITL is the interval between content chunks.
Reported energy is total board input power over measured wall time divided by
output tokens; it is not incremental energy above idle.

## 1.5B Performance

| Topology | Prompt | TTFT p50 | ITL p50 | Request p50 | Output tok/s | Mean power | J/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One node | Short | 303 ms | 24 ms | 2,128 ms | 30.07 | 14.34 W | 0.477 |
| Two nodes | Short | 411 ms | 31 ms | 2,786 ms | 22.97 | 20.18 W | 0.878 |
| One node | Medium | 548 ms | 24 ms | 2,386 ms | 26.81 | 15.32 W | 0.571 |
| Two nodes | Medium | 954 ms | 31 ms | 3,390 ms | 18.86 | 19.78 W | 1.049 |
| One node | Long | 1,282 ms | 24 ms | 3,148 ms | 20.32 | 16.24 W | 0.799 |
| Two nodes | Long | 2,581 ms | 31 ms | 5,179 ms | 12.35 | 19.45 W | 1.574 |

All six scenarios completed 30/30 requests. P95 values were close to P50:
the largest P50-to-P95 request-latency spread was 29 ms.

Relative to one node, two-node execution had:

- 29.2% higher median ITL for every prompt size;
- 23.6%, 29.7%, and 39.2% lower aggregate output throughput;
- 35.6%, 74.1%, and 101.3% higher TTFT as prompt size increased;
- 1.84x, 1.84x, and 1.97x energy per output token.

The increasing TTFT gap tracks activation volume. Grumpy's wired RX+TX delta
over each telemetry window, including warmups and protocol traffic, was
24.6 MB, 105.8 MB, and 349.2 MB for short, medium, and long prompts.

## 25W Versus MAXN_SUPER

The performance cases were repeated after correcting Grumpy's boot-time
`nvpmodel` ordering so both nodes used the same TPC mask (`240`). The 25W mode
ran at 1,344 MHz CPU and 918 MHz GPU; MAXN_SUPER ran at 1,728 MHz CPU and
1,020 MHz GPU. Both used 3,199 MHz EMC and `jetson_clocks`.

| Scenario | Runs | 25W p50 | MAXN p50 | Latency change | 25W tok/s | MAXN tok/s | Power change | J/token change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B, one node, short | 30 | 2,392 ms | 2,234 ms | -6.6% | 26.75 | 28.63 | +8.9% | +1.7% |
| 1.5B, one node, long | 20 | 3,506 ms | 3,246 ms | -7.4% | 18.27 | 19.71 | +10.1% | +2.0% |
| 1.5B, two nodes, short | 30 | 2,947 ms | 2,726 ms | -7.5% | 21.69 | 23.50 | +7.0% | -1.3% |
| 1.5B, two nodes, long | 20 | 5,767 ms | 5,148 ms | -10.7% | 11.10 | 12.42 | +7.7% | -3.7% |
| 14B, two nodes, 16 tokens | 10 | 2,444 ms | 2,303 ms | -5.8% | 6.54 | 6.95 | +6.7% | +0.5% |

MAXN_SUPER reduced TTFT by 4.6% to 12.8% and median ITL by 5.4% to 8.8%.
Its clearest result was the long distributed 1.5B case: 10.7% lower
request latency and 11.9% higher throughput. Mean cluster power rose from
18.13 W to 19.53 W, but faster completion reduced measured energy per token
by 3.7%.

The efficiency result is not universal. One-node energy per token increased
about 2%, and the 14B change was effectively flat at +0.5%. MAXN_SUPER is
therefore justified for latency-sensitive experiments; 25W remains the better
default when power is constrained. Maximum GPU temperature across this A/B
was 61.7 C, with no observed thermal throttling.

## Correctness And Residency

For the 1.5B model, one-node and two-node greedy generation both returned token
IDs `[11, 1052, 572, 264]`, or `", there was a"`. The distributed trace showed:

- Dopey owned layers `[0, 14)` and Grumpy owned `[14, 28)`;
- each decode step transferred one 6,144-byte F32 hidden activation;
- each receiving-stage input CRC matched the sending-stage output CRC;
- Dopey resident weights: 532,897,792 bytes and 169 tensors;
- Grumpy resident weights: 578,472,448 bytes and 170 tensors;
- the two partitions summed exactly to 1,111,370,240 tensor bytes and 339
  tensors.

## Concurrency

At concurrency 2, aggregate throughput remained effectively unchanged while
median request latency doubled:

| Topology | C1 p50 | C2 p50 | C1 tok/s | C2 tok/s |
| --- | ---: | ---: | ---: | ---: |
| One node | 2,128 ms | 4,287 ms | 30.07 | 29.85 |
| Two nodes | 2,786 ms | 5,589 ms | 22.97 | 22.87 |

ITL remained 24 ms on one node and 31 ms on two nodes. The current runtime
serializes or queues sessions; it does not gain throughput from two concurrent
requests.

## Q4_K_M Model Scaling

The controlled scaling sweep used the same coding prompt, a 1,024-token
context, 64 output tokens, three warmups, 20 measured requests, concurrency 1,
and MAXN_SUPER on both nodes. All scenarios completed 20/20 requests.

| Model | Placement | Load | TTFT p50 | ITL p50 | Request p50 | Output tok/s | Mean power | J/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | One node | 4.39 s | 103 ms | 24 ms | 1,683 ms | 37.51 | 16.96 W | 0.452 |
| 1.5B | Two nodes | 4.74 s | 120 ms | 31 ms | 2,141 ms | 29.83 | 21.73 W | 0.728 |
| 3B | One node | 6.21 s | 126 ms | 41 ms | 2,809 ms | 22.43 | 18.79 W | 0.838 |
| 3B | Two nodes | 7.66 s | 156 ms | 49 ms | 3,316 ms | 19.27 | 23.96 W | 1.244 |
| 7B | One node | 15.27 s | 248 ms | 79 ms | 5,308 ms | 12.05 | 20.70 W | 1.718 |
| 7B | Two nodes | 20.54 s | 241 ms | 87 ms | 5,819 ms | 10.99 | 26.34 W | 2.396 |
| 14B | Two nodes | 35.25 s | 418 ms | 159 ms | 10,522 ms | 6.08 | 27.24 W | 4.479 |

The two-node throughput penalty narrowed as model size increased: 20.5% for
1.5B, 14.1% for 3B, and 8.8% for 7B. Distribution still increased cluster
energy per token by 61.1%, 48.5%, and 39.4%, respectively. The 14B Q4_K_M
model was two-node only: its 8,982,142,976 resident tensor bytes exceed one
node's physical RAM.

The 7B model loaded on one node at this sweep's 1,024-token context. A
4,096-token request failed when llama.cpp also needed a 2,432 MiB CUDA compute
buffer. Model-file capacity and usable model-plus-context capacity are
therefore separate boundaries.

## HumanEval

The coding evaluation used EvalPlus HumanEval and HumanEval+, one greedy sample
for each of 164 tasks, a 1,536-token context, and at most 512 output tokens.
The 1.5B, 3B, and 7B models ran on Dopey; 14B ran as a two-node pipeline.
Scoring ran in an 8 GiB Docker container with networking disabled.

| Model | Placement | HumanEval pass@1 | HumanEval+ pass@1 | Fixed-64 tok/s |
| --- | --- | ---: | ---: | ---: |
| 1.5B | One node | 118/164 (72.0%) | 112/164 (68.3%) | 37.51 |
| 3B | One node | 134/164 (81.7%) | 126/164 (76.8%) | 22.43 |
| 7B | One node | 146/164 (89.0%) | 138/164 (84.1%) | 12.05 |
| 14B | Two nodes | 147/164 (89.6%) | 142/164 (86.6%) | 6.08 |

Wilson 95% confidence intervals overlap for 7B and 14B. Paired exact McNemar
tests also did not distinguish them: HumanEval `p=1.0`, HumanEval+
`p=0.481`. The measured 14B score increase is therefore not a supported
quality win on this 164-task sample. The 1.5B-to-3B and 3B-to-7B paired
improvements had unadjusted `p<0.025`; those values remain subject to
multiple-comparison and single-run limitations.

This result makes 7B the local Pareto choice when it fits: it retained about
twice the fixed-output throughput and used 38% of the energy per output token
of the 14B split, with no statistically distinguishable HumanEval loss. The
14B result instead proves that JetsonFabric can expose otherwise unavailable
aggregate model capacity. Harder or less saturated evaluations are needed to
measure whether that capacity produces a useful quality gain.

## 14B Capacity Boundary

This earlier capacity experiment used Q4_0 and is separate from the Q4_K_M
scaling and HumanEval runs above. The Qwen2.5-Coder 14B Q4_0 GGUF is
8,517,725,568 bytes. A one-node deployment failed cleanly after 16.13 seconds
when CUDA attempted a 7,699.79 MiB model buffer with only about 6.58 GiB
available. There was no kernel OOM kill.

A two-node deployment loaded in 34.34 seconds:

| Node | Layers | Resident tensor bytes | Tensors |
| --- | --- | ---: | ---: |
| Dopey | `[0, 24)` | 4,155,506,688 | 289 |
| Grumpy | `[24, 48)` | 4,356,251,648 | 290 |

The partitions totaled 8,511,758,336 tensor bytes and 579 tensors. The
difference from file size is GGUF metadata. The physical trace completed and
maintained CRC continuity; each decode activation was 20,480 bytes.

At a 1,024-token context:

| Prompt | Success | TTFT p50 | ITL p50 | Request p50 | Output tok/s | Mean power | J/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 10/10 | 427 ms | 122 ms | 8,213 ms | 7.77 | 27.45 W | 3.531 |
| Medium | 10/10 | 2,975 ms | 122 ms | 10,825 ms | 5.91 | 24.88 W | 4.209 |

A 4,096-token deployment could load only after explicitly unloading the old
deployment. Its requests then failed because Grumpy could not allocate a
768 MiB KV cache plus a 2,456 MiB prefill compute buffer beside its model
partition. This is the current practical context boundary, not a model-file
load failure.

An in-place 1K-to-4K switch also failed because the transactional switch keeps
the old epoch resident until the replacement is ready. A cold 4K load
succeeded, showing that zero-downtime replacement needs explicit memory
headroom or an allowed disruptive-switch mode.

## Failure And Recovery

Killing Grumpy during a 512-token stream caused the client to fail in 2.149
seconds with `runtime_stage_unreachable`; the process and coordinator did not
hang. After lease expiry, the deployment correctly became `degraded`.

Recovery is not yet correct. When Grumpy restarted:

1. membership returned and the coordinator reported the deployment `active`;
2. Grumpy's runtime remained idle with no loaded deployment;
3. the next request failed with `no_active_deployment`;
4. an explicit deployment switch restored service in 4.074 seconds.

The coordinator must verify stage runtime identity/state before returning an
active phase and must reload the last intent on a restarted runtime.

## UTF-8 Transport Regression

The first 14B HumanEval run exposed a distributed transport bug at task 72:
one tokenizer piece ended in byte `0x9e`, which is not valid as an independent
UTF-8 JSON string. Tokenizer pieces are byte fragments and may split a Unicode
code point even though their concatenation is valid text.

Stagewire now uses a numeric `message_bytes` field for non-UTF-8 token
fragments while retaining the existing `message` field for valid text.
`GenerationRunner` buffers fragments and emits only complete UTF-8 prefixes at
the OpenAI SSE boundary. Unit tests cover exact byte round trips and a
three-token split Unicode code point. On the physical CUDA cluster, the
original 128-token failure then passed, and a 512-token regression completed
with a natural stop after 412 tokens.

The final 14B score combines deterministic samples 0-71 from the original run
with samples 72-163 generated after the fix. Its reported HumanEval telemetry
covers only the post-fix tasks 72-163.

## Thermal And Measurement Notes

- Maximum GPU temperature in the original performance suite was 63.1 C.
- The longer HumanEval runs reached 68.8 C on the active node.
- No thermal throttling was observed.
- The 14B 4K failure window reached 7,305 MB RAM and 214 MB swap.
- Wi-Fi was disabled after an initial route audit found traffic for Grumpy's
  wired address traversing Wi-Fi. All reported distributed runs were repeated
  over Ethernet.
- The original MAXN sweep had a TPC-mask asymmetry because
  `nvidia-cdi-refresh` initialized Grumpy's GPU before `nvpmodel`. The
  power-mode A/B above was run only after ordering CDI refresh after
  `nvpmodel` and verifying mask `240` and the expected clocks on both nodes.
- A stock llama.cpp CUDA/RPC control was attempted but excluded: compiling the
  full CUDA template set on the Orins had not completed after approximately
  27 minutes. No native/RPC performance claim is made.

## Artifacts

Raw local artifacts are under `data/benchmarks/m7-20260727/` and remain
gitignored. Power-mode A/B artifacts are under
`data/benchmarks/power-20260727/`. Model-scaling and HumanEval artifacts are
under `data/benchmarks/model-scaling-20260727/` and
`data/benchmarks/humaneval-20260727/`. The local normalized outputs are:

- `data/benchmarks/m7-20260727/summary.csv`
- `data/benchmarks/m7-20260727/summary.json`
- `data/benchmarks/power-20260727/summary.json`
- `data/benchmarks/model-scaling-20260727/summary.csv`
- `data/benchmarks/model-scaling-20260727/summary.json`
- `data/benchmarks/humaneval-20260727/humaneval-summary.csv`
- `data/benchmarks/humaneval-20260727/humaneval-summary.json`

Publication inputs and generated figures are tracked under
`docs/benchmarks/data/` and `docs/benchmarks/figures/`. Regenerate the figures
with `python3 tools/bench/plot_results.py`.

The benchmark client now preserves arbitrary request fields and reports
streaming TTFT, ITL distributions, and partial-stream errors.
