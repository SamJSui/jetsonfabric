# Two-Orin Serving Matrix

Date: 2026-07-29

Base commit: `6647dea` (`Refactor runtime and coordinator boundaries`)

Candidate: local `feat/activation-encoding` worktree. Commit the tested
implementation and preserve the raw reports before treating these values as
published benchmark provenance.

## Question

How effectively can two Jetson Orin Nano 8 GB nodes serve Qwen2.5-Coder models
that fit on one node, and what changes when the model requires both nodes?

## Environment

- two Jetson Orin Nano 8 GB developer kits in `MAXN_SUPER`;
- wired 1 GbE through the same router;
- Qwen2.5-Coder Instruct Q4_K_M models;
- 1,536-token context, greedy sampling, and 128 requested output tokens;
- F16 activation transport for pipeline deployments;
- two warmup requests before every measured suite;
- identical node and runtime-worker binaries on both hosts.

The benchmark client sends OpenAI-compatible streaming requests and derives
TTFT and ITL from runtime token events. It verifies event counts against final
usage. Aggregate output throughput is the total generated tokens divided by
suite wall-clock time, not the speed of one stream.

## Replica Suites

These models fit on one 8 GB node. `single` sends all requests to one node;
`replicas` distributes requests across independent full-model copies.

| Model | Suite | Success | Output tok/s | Request/s | TTFT p50 | ITL p50 | E2E p50 | E2E p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | Single C1 | 10/10 | 40.20 | 0.314 | 73 ms | 24 ms | 3.16 s | 3.34 s |
| 1.5B | Single C2 | 10/10 | 40.44 | 0.316 | 94 ms | 48 ms | 6.18 s | 6.61 s |
| 1.5B | Replicas C2 | 10/10 | **80.06** | 0.626 | 75 ms | 24 ms | 3.19 s | 3.22 s |
| 1.5B | Replicas C4 | 12/12 | 79.99 | 0.625 | 93 ms | 46 ms | 5.39 s | 12.32 s |
| 3B | Single C1 | 10/10 | 23.34 | 0.182 | 114 ms | 42 ms | 5.51 s | 5.53 s |
| 3B | Single C2 | 10/10 | 23.36 | 0.183 | 152 ms | 42 ms | 6.03 s | 47.09 s |
| 3B | Replicas C2 | 10/10 | **46.45** | 0.363 | 115 ms | 42 ms | 5.48 s | 5.61 s |
| 3B | Replicas C4 | 12/12 | 46.57 | 0.364 | 152 ms | 42 ms | 9.00 s | 15.49 s |
| 7B | Single C1 | 10/10 | 12.35 | 0.096 | 198 ms | 80 ms | 10.36 s | 10.43 s |
| 7B | Single C2 | 10/10 | 12.36 | 0.097 | 271 ms | 80 ms | 12.48 s | 62.57 s |
| 7B | Replicas C2 | 10/10 | **24.63** | 0.192 | 198 ms | 80 ms | 10.36 s | 10.47 s |
| 7B | Replicas C4 | 12/12 | 24.63 | 0.192 | 278 ms | 80 ms | 16.10 s | 37.25 s |

Two replicas nearly double aggregate throughput at concurrency 2 because each
request owns one GPU. Concurrency 4 does not add throughput and worsens queueing
latency. The high single-node C2 p95 values for 3B and 7B are isolated tail
outliers in ten-request samples; they are retained rather than hidden.

## Pipeline Suites

Pipeline deployments hold one contiguous layer partition on each node.
Allocations were chosen by measuring decode-stage execution time, then moving
layers until the stages were approximately balanced.

| Model | Layers | Suite | Success | Output tok/s | Request/s | TTFT p50 | ITL p50 | E2E p50 | E2E p95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 18/10 | Pipeline C1 | 10/10 | 29.55 | 0.231 | 154 ms | 32 ms | 4.34 s | 4.43 s |
| 1.5B | 18/10 | Pipeline C2 | 10/10 | **63.44** | 0.496 | 148 ms | 29 ms | 4.02 s | 4.09 s |
| 1.5B | 18/10 | Pipeline C4 | 12/12 | 63.79 | 0.498 | 4.13 s | 29 ms | 7.96 s | 8.13 s |
| 3B | 21/15 | Pipeline C1 | 10/10 | 18.60 | 0.145 | 210 ms | 51 ms | 6.89 s | 6.95 s |
| 3B | 21/15 | Pipeline C2 | 10/10 | **39.77** | 0.311 | 203 ms | 48 ms | 6.44 s | 6.49 s |
| 3B | 21/15 | Pipeline C4 | 12/12 | 39.81 | 0.311 | 6.60 s | 47 ms | 12.78 s | 12.95 s |
| 7B | 16/12 | Pipeline C1 | 10/10 | 11.54 | 0.090 | 314 ms | 85 ms | 11.16 s | 11.26 s |
| 7B | 16/12 | Pipeline C2 | 10/10 | **22.60** | 0.177 | 316 ms | 85 ms | 11.33 s | 11.47 s |
| 7B | 16/12 | Pipeline C4 | 12/12 | 22.56 | 0.176 | 11.75 s | 86 ms | 22.73 s | 22.98 s |
| 14B | 26/22 | Pipeline C1 | 6/6 | 6.27 | 0.049 | 517 ms | 155 ms | 20.38 s | 20.61 s |
| 14B | 26/22 | Pipeline C2 | 6/6 | **12.19** | 0.095 | 533 ms | 159 ms | 20.99 s | 21.15 s |

Concurrency 2 is the useful operating point. It overlaps different requests
across the two stages. Concurrency 4 adds queueing without increasing aggregate
throughput.

For the 14B capacity case, concurrency 2 increased aggregate throughput by
94.5% over the same pipeline at concurrency 1, while median TTFT rose 3.1%,
ITL rose 2.6%, and E2E latency rose 3.0%. A one-node 14B comparison is not
possible because its 8.98 GB of resident tensor payloads exceed one board's
physical RAM.

## Pipeline Efficiency

This table compares the best pipeline C2 result with the measured alternatives
for models that fit on one node.

| Model | Pipeline vs one-node C1 | Pipeline throughput retained vs replicas C2 |
| --- | ---: | ---: |
| 1.5B | +57.8% | 79.2% |
| 3B | +70.4% | 85.6% |
| 7B | +83.0% | 91.8% |

The pipeline penalty relative to two replicas shrinks from 20.8% at 1.5B to
8.2% at 7B. Larger models spend more time computing each stage, so the roughly
fixed per-hop overhead becomes a smaller fraction of each token.

## Runtime Timing

The candidate reports engine execution, activation codec time, total stage
time, and caller-observed remote overhead. Remote overhead includes queue wait,
serialization, HTTP, and network time; it is not a pure wire measurement.
Values below are C2 decode-call averages.

| Model | Stage 0 execution | Stage 1 execution | F16 encode | F16 decode | Remote overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 12.58 ms | 12.61 ms | 0.043 ms | 0.022 ms | 4.50 ms |
| 3B | 21.61 ms | 21.72 ms | 0.055 ms | 0.027 ms | 4.67 ms |
| 7B | 40.66 ms | 40.23 ms | 0.091 ms | 0.044 ms | 4.89 ms |
| 14B | 75.71 ms | 76.45 ms | 0.127 ms | 0.061 ms | 7.14 ms |

The codec itself is not the bottleneck. The next transport optimization should
target request framing, queueing, and the HTTP hop, then remeasure before
introducing a more complex protocol.

## Quality And Useful Work

The prior EvalPlus run used one greedy sample for each of 164 HumanEval tasks.
Correct completions per minute combines measured pass count with end-to-end
suite duration; it is not a standardized EvalPlus metric.

| Model | HumanEval+ pass@1 | Correct tasks | Duration | Correct completions/min |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 68.3% | 112/164 | 14.2 min | 7.887 |
| 3B | 76.8% | 126/164 | 27.5 min | 4.582 |
| 7B | 84.1% | 138/164 | 49.9 min | 2.766 |
| 14B | 86.6% | 142/164 | 77.5 min | 1.831 |

The 7B model remains the measured quality/performance knee. The 14B model adds
four passing tasks in this sample but requires both nodes and substantially
more generation time. The 7B-to-14B quality difference was not statistically
distinguishable in the original analysis.

## Operational Findings

- All 234 measured serving requests completed successfully.
- The physical distributed CUDA validator preserved exact greedy tokens,
  activation CRC continuity, two CUDA-active hosts, and partial model
  residency after the telemetry changes.
- The 14B 26/22 split held 4,700,495,872 resident weight bytes on stage 0 and
  4,281,647,104 bytes on stage 1.
- Switching directly between large 14B partitions could not keep both old and
  new allocations resident during prepare. A cold idle restart loaded 26/22
  successfully. Steady-state fit therefore does not imply enough hot-switch
  headroom on an 8 GB node.

## Claim Boundary

The matrix demonstrates capacity expansion, near-linear aggregate throughput
for two concurrent 14B requests within the pipeline, and decreasing pipeline
overhead as model size grows. It does not demonstrate faster single-request
decode than one node, production stability, an official GenAI-Perf result, or
an automatic stage-balancing planner.
