# Phase 1: colocated pipeline validation

Phase 1 proves the complete JetsonFabric software path before a second physical
Jetson is available:

```text
one physical Linux host
  -> two logical jetsonfabric-node processes
  -> two supervised C++ runtime workers
  -> two different transformer layer ranges
  -> real activation handoff
  -> persistent decode state
  -> OpenAI-compatible chat response
```

This is a compute-partitioning and orchestration proof. Each runtime still loads
the complete GGUF; model-weight memory partitioning is a later milestone.

## Prerequisites

- Go, CMake, a C++20 compiler, curl, jq, awk, and sha256sum;
- enough memory to load the selected GGUF twice;
- a Llama or Qwen2-family GGUF supported by the pinned llama.cpp patch.

The intended first local model is Qwen2.5-Coder 1.5B Q4.

## One-command validation

From the repository root:

```bash
MODEL_PATH=/absolute/path/to/qwen2.5-coder-1.5b-q4.gguf \
MODEL_ID=qwen2.5-coder-1.5b-q4 \
bash scripts/local/validate-colocated-pipeline.sh
```

The script:

1. builds the pinned patched llama.cpp runtime and Go node;
2. hashes the GGUF with SHA-256;
3. derives the real model layer count;
4. starts two logical nodes on one host;
5. confirms both advertise the same engine, model ID, artifact hash, execution
   mode, and complementary stage assignments; it also records each backend;
6. compares two distributed greedy tokens with a one-runtime baseline;
7. verifies activation byte and CRC continuity for prefill and decode;
8. sends an ordinary `POST /v1/chat/completions` request to the follower node;
9. verifies leader forwarding, coordinator planning, and OpenAI response shape;
10. writes `reports/phase1-colocated.json`.

Useful overrides:

```bash
JF_SKIP_BUILD=true                 # reuse existing binaries
JF_MAX_TOKENS=2
JF_CTX_SIZE=4096
JF_RUNTIME_THREADS=4
JF_REPORT_PATH=/tmp/jf-phase1.json
JF_NODE0_PORT=19180
JF_NODE1_PORT=19181
```

## Passing report

The JSON report contains:

- model path, ID, layer count, split point, and SHA-256;
- membership and topology evidence;
- baseline and distributed token IDs;
- activation CRC continuity;
- per-stage engine latency;
- prefill and decode activation sizes;
- total OpenAI request duration;
- the final OpenAI-compatible response.

Phase 1 is complete when the report says:

```text
topology                  = colocated
stage_count               = 2
logical_node_count        = 2
physical_host_count       = 1
tokens_match              = true
activation_crc_continuity = true
```

and both members advertise the same non-empty `runtime_model_sha256`.

## Compatibility and placement metadata

Every supervised node advertises:

```text
runtime_engine
runtime_model_id
runtime_model_sha256
runtime_compute_backend
runtime_execution_mode
runtime_stage_index
runtime_stage_count
runtime_layer_start
runtime_layer_end
```

The coordinator currently groups candidates for correctness by engine adapter,
model ID, exact artifact SHA-256, and pipeline execution mode. Compute backend is
still advertised, but CPU versus CUDA is placement and telemetry metadata rather
than part of activation compatibility.

The C++ `StageWorker` independently rejects a request whose `model_id`, node
name, stage position, or layer range differs from that runtime's configured
assignment.

## What Phase 1 does not prove

Phase 1 does not prove:

- two physical hosts;
- LAN activation transfer;
- CUDA was compiled and actually used;
- reduced per-node model memory;
- dynamic model loading or deployment rebalancing;
- pipeline overlap or throughput microbatching.

Those are intentionally left for the physical and optimization phases.
