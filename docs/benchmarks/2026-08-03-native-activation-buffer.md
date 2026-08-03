# Native Activation Ownership and StageWire Writes

Date: 2026-08-03

Status: physical validation complete

Final revision: `90fdbee`

Baseline revision: `455ab56`

## Change

The native Qwen2 executor now returns one byte-backed activation allocation.
That allocation moves through `NativeStageExecutor`, `StageWorker`,
`ModelManager`, and `GenerationRunner` instead of being copied through an
intermediate `std::vector<float>`. Direct HTTP transport builds the same
StageWire v2 frame as a metadata prefix plus activation payload and sends those
segments in one `sendmsg()` operation.

This is not an end-to-end zero-copy path. Incoming HTTP parsing and StageWire
decoding still materialize receive storage, and the runtime does not yet use a
pinned receive-buffer pool.

## Physical Gate

The matched run used two 8 GB Jetson Orin Nano Super nodes in `MAXN_SUPER`,
full-duplex Gigabit Ethernet, the native Qwen2 CUDA engine, F32 activations,
F16 KV cache, and direct runtime HTTP transport. Each row used two warmups and
10 measured streaming requests with 128 output tokens.

Both models returned the expected greedy sequence `[11, 1052, 572, 264]` in
the correctness gate. All four performance rows completed 10/10 requests.
Resident weights remained exactly partitioned:

| Model | Dopey layers / bytes | Grumpy layers / bytes | Total bytes |
|---|---:|---:|---:|
| Qwen2.5-Coder 3B Q4 | 0-21 / 1,139,644,416 | 21-36 / 959,332,352 | 2,098,976,768 |
| Qwen2.5-Coder 14B Q4 | 0-26 / 4,700,495,872 | 26-48 / 4,281,647,104 | 8,982,142,976 |

## Results

The baseline is the prior native scaling run under the same workload. Remote
overhead is the runtime's mean decode remote-call time minus stage execution
time.

| Model | Concurrency | Baseline tok/s | Final tok/s | Delta | Final TTFT p50 | Final ITL p50 | Final latency p50 | Remote overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B | 1 | 22.19 | 22.44 | +1.12% | 120 ms | 43 ms | 5,703 ms | 1.262 -> 1.119 ms |
| 3B | 2 | 45.00 | 45.41 | +0.91% | 117 ms | 43 ms | 5,637 ms | 1.288 -> 1.111 ms |
| 14B | 1 | 6.40 | 6.42 | +0.28% | 400 ms | 153 ms | 19,929 ms | 2.003 -> 1.723 ms |
| 14B | 2 | 12.51 | 12.51 | -0.02% | 398 ms | 156 ms | 20,507 ms | 2.246 -> 1.937 ms |

The change reduced measured decode remote overhead by 11.3% to 14.0% without
changing tokens, residency, or StageWire bytes. End-to-end throughput remained
within roughly 1% of baseline because GPU stage execution dominates these
workloads. This is a transport and ownership cleanup, not evidence of a large
model-speedup.

An intermediate physical build sent the HTTP headers, StageWire prefix, and
payload with three separate `send()` calls. On the 3B decode path this triggered
about 44 ms of remote overhead and halved throughput. The final response path
uses one segmented `sendmsg()` call, restoring the baseline while retaining the
non-flattened payload.

The diagnostic artifact's overall status is `failed` because its later 14B
switch overlapped a superseded benchmark process that was still holding model
memory. Only its completed 3B rows are used for the transport diagnosis.

## Evidence

- [Final 3B/14B matrix](evidence/20260803-native-activation-buffer/native-activation-buffer-90fdbee.json)
- [Intermediate 3B diagnostic](evidence/20260803-native-activation-buffer/native-activation-buffer-89b7939-diagnostic.json)
- [Prior native scaling baseline](evidence/20260803-native-scaling/)
