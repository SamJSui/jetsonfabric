# Two-Orin Concurrent Serving Optimization

Date: 2026-07-28

Base commit: `6647dea` (`Refactor runtime and coordinator boundaries`)

Candidate: local `feat/activation-encoding` worktree. The final commit hash
must replace this line before publishing benchmark claims.

## Question

Does a two-node pipeline provide a serving benefit over one Jetson when both
deployments use the same model, context size, prompt, and load?

## Environment

- two Jetson Orin Nano 8 GB developer kits;
- `MAXN_SUPER` power mode on both nodes;
- wired 1 GbE through the same router;
- Qwen2.5-Coder 1.5B Instruct Q4_K_M;
- model SHA-256
  `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`;
- 4,096-token context;
- greedy sampling;
- F16 activation transport for the two-node deployment;
- measured 18-layer/10-layer stage allocation;
- OpenAI-compatible streaming chat endpoint.

The short workload naturally stopped after 20 output tokens. The long workload
used `examples/genai-perf-request.json` and reached its 128-token limit.
Two or three requests warmed each deployment before measurement. The short
test used 20 measured requests; the long test used 10.

The benchmark client used for this run derived output-token counts from
nonempty SSE content chunks. Every successful long-output request in this fixed
English workload produced 128 such chunks and stopped with
`finish_reason=length`, so the aggregate calculation used 1,280 output chunks.
That is sufficient to reproduce this result but is not a general substitute
for authoritative model token counts. The current candidate now emits a
runtime token index on every token event, including empty UTF-8 fragments, and
final prompt/completion usage; future runs reject streams whose event count and
authoritative usage disagree.

## Headline Result

The two-node Qwen2.5-Coder 1.5B deployment completed all 10 long-output
requests at concurrency 2. It sustained 62.67 aggregate output tok/s and 0.490
requests/s, with median TTFT of 161 ms, median ITL of 29 ms, and median
end-to-end latency of 4,062 ms.

Aggregate output throughput is the total number of output tokens from both
concurrent requests divided by benchmark wall-clock time. It is not per-request
token generation speed. The raw result demonstrates that the two physical
Jetsons can remain productively occupied by separate requests in the same
pipeline.

## Changes

1. Runtime HTTP stage connections remain open instead of reconnecting for
   every stage invocation.
2. Activation encoding is selected through the registered F32/F16 codec
   interface.
3. Deployment policy accepts validated `stage_layer_counts`; equal allocation
   remains the default.
4. Llama contexts retain the requested KV context size but size their compute
   batch from the actual prefill token count instead of the full context size.
5. The benchmark output reports request throughput, output-token throughput,
   mean output length, and p90/p99 latency fields.
6. Streaming benchmark accounting uses runtime token events and verifies them
   against final authoritative usage.

The fourth change removed a 2.398 GiB CUDA compute-buffer allocation per
session observed with a 4,096-token context. Before the fix, one-node
concurrency 2 failed 9 of 10 long requests with CUDA out-of-memory errors.

## Results

All values below came from successful requests. Throughput is total output
tokens divided by wall-clock benchmark duration. Latency columns are medians.

| Output | Placement | Concurrency | Success | Output tok/s | Request/s | TTFT | ITL | E2E |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | One node | 1 | 20/20 | 32.85 | 1.642 | 85 ms | 24 ms | 609 ms |
| 20 | One node | 2 | 20/20 | 32.59 | 1.630 | 106 ms | 48 ms | 1,157 ms |
| 20 | Two nodes | 1 | 20/20 | 22.46 | 1.123 | 157 ms | 32 ms | 891 ms |
| 20 | Two nodes | 2 | 20/20 | 43.26 | 2.163 | 158 ms | 29 ms | 912 ms |
| 128 | One node | 1 | 10/10 | 39.99 | 0.312 | 83 ms | 24 ms | 3,186 ms |
| 128 | One node | 2 | 10/10 | 39.88 | 0.312 | 103 ms | 48 ms | 6,136 ms |
| 128 | Two nodes | 1 | 10/10 | 29.49 | 0.230 | 171 ms | 32 ms | 4,339 ms |
| 128 | Two nodes | 2 | 10/10 | 62.67 | 0.490 | 161 ms | 29 ms | 4,062 ms |

At concurrency 2:

- short-output aggregate throughput improved 32.7% over one node;
- long-output aggregate throughput improved 57.1% over one node;
- short-output median E2E latency fell 21.2%;
- long-output median E2E latency fell 33.8%.

At concurrency 1, one node remained faster. Pipeline parallelism increases the
latency of one isolated request because every token executes both stages and
crosses the network.

## Stage Balance

The equal 14/14 split averaged 11.6 ms for stage 0 and 17.8 ms for stage 1
during decode. Moving four layers to stage 0 produced an 18/10 split averaging
15.1 ms and 15.5 ms. This lets separate requests occupy the two GPUs at the
same time instead of waiting on a consistently slower final stage.

The inflection point is not known before measurement. The final stage also
owns final normalization, output projection, and sampling, so equal layer
counts do not imply equal work. The practical search starts at 14/14, warms the
deployment, measures each stage's decode latency, and moves layers away from
the slower stage until the larger stage latency stops falling. For this model
and hardware, the measured stage-latency curves crossed near 18/10.

The split is explicit benchmark input, not an automatic planner claim. It may
change with the model, context, power mode, runtime revision, or hardware.

## Correctness

The physical distributed CUDA validator passed after each final change. The
two-node deployment retained:

- two distinct CUDA-active physical hosts;
- exact expected greedy tokens `[11, 1052, 572, 264]`;
- continuous activation CRCs between stages;
- partial resident model weights on both nodes;
- the active deployment identity and epoch.

F16 is lossy. Four-token equivalence is a regression gate, not a quality
evaluation. F16 should not become the default until a broader accuracy suite
passes.

## Artifacts And Provenance

The benchmark report records the tested configuration, aggregate results, and
sample counts. The candidate still exists only in the local dirty worktree.
Before publishing these values as repository evidence:

1. commit the tested implementation and replace the candidate placeholder with
   its commit hash;
2. preserve the benchmark client's per-request JSON output for every row;
3. record the deployment response, runtime identity, model-memory response,
   and physical CUDA validation output alongside those samples;
4. rerun the same commands from the committed revision.

## Claim Boundary

These results demonstrate a concurrent serving throughput benefit for one
model and two clients. They do not establish lower latency for a single
request, universal scaling across models, production stability, or an
official NVIDIA GenAI-Perf submission. The built-in harness uses GenAI-Perf
metric definitions and preserves per-request samples; an official tool run is
still required before using the GenAI-Perf name in an external claim.
Per-request artifacts and the final candidate commit hash must also be checked
in before treating this report as published benchmark provenance.
