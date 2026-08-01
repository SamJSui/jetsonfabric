# Tensor-Parallel Feasibility On Two Orin Nanos

This experiment asks a narrow question: can two 8 GB Jetson Orin Nanos perform
fine-grained tensor-parallel work quickly enough over their existing Gigabit
Ethernet link to justify a full runtime implementation?

The answer is **yes for further development, not yet for full-model serving
claims**. A real two-rank CUDA SwiGLU MLP sublayer nearly halved both median and
p95 latency for the Qwen2.5-Coder 14B shape. This proof does not yet execute a
complete transformer block or generate tokens.

## Setup

- two Jetson Orin Nano 8 GB developer kits: Dopey and Grumpy;
- JetPack 7.2 / Jetson Linux R39.2;
- MAXN_SUPER on both nodes with `jetson_clocks` locked;
- direct wired Gigabit Ethernet through the same router;
- CUDA SM87, FP16 cuBLAS GEMMs, and pinned host buffers;
- persistent TCP, with one FP16 partial output sent by each rank;
- 50 warmup passes followed by 500 measured passes per MLP configuration.

The benchmark implements the Qwen-style SwiGLU path: gate projection, up
projection, SiLU multiplication, and down projection. The local case owns the
full intermediate width. In the distributed case each rank owns half of that
width, computes a partial down-projection output, copies it to pinned host
memory, exchanges it over TCP, sums both partials, and copies the reduced output
back to CUDA memory.

## Network Lower Bound

Both NICs negotiated at 1 Gbit/s. A ten-second `iperf3` run sustained 938.2
Mbit/s with zero retransmits. Before clocks were locked, 100 unloaded ICMP
samples measured 0.955 ms average round-trip latency.

The application-level benchmark below uses persistent TCP and performs an
actual two-rank float sum. The payloads match FP16 hidden vectors even though
the microbenchmark uses float elements to perform the host sum.

| Profile | Bytes/rank | Iterations | p50 | p95 | Sequential sync projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B hidden vector | 3,072 | 1,000 | 196 us | 201 us | 11.00 ms/token at 56 syncs |
| 14B hidden vector | 10,240 | 1,000 | 178 us | 230 us | 17.05 ms/token at 96 syncs |

The projection assumes two collectives per transformer block and no overlap.
It is a communication lower bound, not a predicted model ITL. The smaller
payload's slightly higher p50 is fixed-cost scheduling noise, not evidence that
larger payloads are inherently faster.

## CUDA MLP Result

Distributed latency uses the slower rank for each percentile. Negative latency
change is an improvement.

| Model shape | Hidden / intermediate | Local p50 | Two-rank p50 | p50 change | Local p95 | Two-rank p95 | p95 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder 1.5B | 1,536 / 8,960 | 1.647 ms | 0.677 ms | **-58.9% (2.43x)** | 1.655 ms | 1.973 ms | +19.2% |
| Qwen2.5-Coder 14B | 5,120 / 13,824 | 8.250 ms | 4.163 ms | **-49.5% (1.98x)** | 8.258 ms | 4.862 ms | **-41.1% (1.70x)** |

For the 14B shape, each rank's median local compute was about 3.88 ms and the
host-staged all-reduce was about 0.29 ms. The collective therefore consumed
only 6.9% of the distributed sublayer's p50 critical path. The result shows
that the larger GEMMs can amortize the existing network and host-staging cost.

The 1.5B median is fast, but its distributed p95 is worse than local. That tail
regression matters because a full decode token repeats synchronization across
every layer. Small-model tensor parallelism is therefore not the first product
target even though this isolated median is favorable.

## What This Proves

- Gigabit Ethernet does not rule out two-rank tensor parallelism on this
  hardware when each synchronization is paired with enough GPU work.
- A Qwen-shaped 14B MLP sublayer can achieve near-linear median scaling and a
  meaningful p95 improvement with no RDMA, NCCL, or new network hardware.
- Tensor-parallel value depends on compute-to-communication ratio; the result
  cannot be generalized from one model size to another.

## What It Does Not Prove

- This is one SwiGLU MLP sublayer, not a complete transformer block.
- It excludes Q/K/V projections, RoPE, KV-cache reads, attention, residuals,
  normalization, embeddings, logits, and sampling.
- It uses dense FP16 cuBLAS matrices rather than llama.cpp's quantized GGUF
  kernels, so its absolute latency is not a full-model prediction.
- It does not yet prove end-to-end token equivalence, TTFT, ITL, throughput,
  memory capacity, or HumanEval quality under tensor parallelism.

## Decision

Proceed with a 14B-first tensor-parallel runtime milestone. The next acceptance
gate is one complete decoder block with two rank synchronizations, exact output
equivalence against an unsharded block, and per-phase CUDA/network telemetry.
Only after that gate should JetsonFabric modify model loading and generation to
shard every block.

The current pipeline runtime remains the supported serving path until a full
tensor-parallel model preserves token equivalence and improves measured
end-to-end ITL or aggregate throughput.

## Reproduction

Build the optional benchmark on each Jetson:

```bash
cmake -S runtime -B runtime/build-tp \
  -DCMAKE_BUILD_TYPE=Release \
  -DJF_BUILD_CUDA_BENCHMARKS=ON \
  -DJF_CUDA_ARCHITECTURES=87
cmake --build runtime/build-tp \
  --target jetsonfabric-tensor-parallel-mlp-bench -j2
```

Run the 14B-shaped baseline on one node:

```bash
runtime/build-tp/jetsonfabric-tensor-parallel-mlp-bench \
  --local --hidden-size 5120 --intermediate 13824 \
  --iterations 500 --warmup 50
```

Start the server rank on one node, then the client rank on the other:

```bash
runtime/build-tp/jetsonfabric-tensor-parallel-mlp-bench \
  --server --listen 0.0.0.0 --port 52510
```

```bash
runtime/build-tp/jetsonfabric-tensor-parallel-mlp-bench \
  --client <server-ip> --port 52510 \
  --hidden-size 5120 --intermediate 13824 \
  --iterations 500 --warmup 50
```
