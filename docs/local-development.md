# Local development

The normal developer path runs one Go node supervising one C++ runtime. That
runtime is stage `0/1` and executes the complete model layer range through the
same pipeline protocol used by multi-node deployments.

Co-located multi-stage execution is test-only. It is intentionally kept out of
`make dev-up` because two runtimes still duplicate process, context, KV, and
compute-buffer overhead even though their model tensor weights are partitioned.

## Configuration

Copy the template and point it at a local GGUF:

```bash
cp .env.example .env.local
```

A Jetson Orin development configuration can use:

```dotenv
MODEL=qwen2.5-coder-1.5b-q4
MODEL_PATH=models/qwen.gguf
JETSONFABRIC_CLUSTER_TOKEN=replace-with-a-random-local-token
RUNTIME_CUDA_ARCH=87
RUNTIME_COMPUTE_BACKEND=cuda
RUNTIME_MODE=pipeline_parallel
RUNTIME_CTX_SIZE=4096
RUNTIME_N_GPU_LAYERS=999

JF_NODE0_PORT=19180
JF_RUNTIME_PORT=19190
JF_DEV_WORK_DIR=.cache/jetsonfabric/dev
```

Every node and supervised runtime in one cluster must receive the same token.
The Makefile supplies a known local-only default when `.env.local` omits it;
persistent or network-accessible clusters must override that value.

`RUNTIME_BUILD_JOBS=1` or `2` is the safest starting point on an 8 GB Jetson.
Use `4` only when compilation is stable and the system is not swapping.

The base stage values may remain:

```dotenv
STAGE_INDEX=0
STAGE_COUNT=1
LAYER_START=0
LAYER_END=28
```

`make dev-up` reads the GGUF's actual layer count and creates the full assignment
`stage 0/1, layers [0, layer_count)` automatically. The configured `LAYER_END`
is used only by lower-level `make run-node` and `make run-runtime` commands.

## Run one development node

From the repository root:

```bash
make test
make dev-up
```

`make dev-up`:

1. acquires an exclusive lock under `.cache/jetsonfabric/dev`;
2. checks that the fixed node and runtime ports are free;
3. loads `.env.local` through the Makefile;
4. builds the selected CUDA or CPU runtime when needed;
5. builds a native `jetsonfabric-node`;
6. reads the model's actual transformer-layer count;
7. starts one Go node and one supervised C++ runtime;
8. records the node and runtime PIDs;
9. configures one full-range pipeline stage;
10. waits for health;
11. sends a real one-token chat request;
12. prints ready only after inference succeeds;
13. remains in the foreground until stopped.

The default endpoints are:

```text
Node:    http://127.0.0.1:19180
Runtime: http://127.0.0.1:19190
```

Both ports are intentionally static for the normal development profile. A second
`make dev-up`, an orphaned runtime, or an unrelated process on either port causes
startup to fail visibly instead of silently selecting another runtime port.

The launcher also holds an exclusive lock. This prevents duplicate launches even
when a developer overrides the default ports.

Logs and lifecycle files are retained under:

```text
.cache/jetsonfabric/dev/
├── dev.lock
├── node.pid
├── runtime.pid
└── logs/
```

## Inspect and use the service

In another terminal:

```bash
make dev-status
```

The status command prints the configured endpoints and recorded PIDs before
querying health, membership, and the one-stage route preview.

Send a completion:

```bash
make dev-chat DEV_PROMPT='Explain why CUDA warps matter.' DEV_MAX_TOKENS=32
```

`make dev-chat` prints the JSON response body even when the server returns an
error. This keeps runtime and pipeline failures visible instead of reducing them
to a generic curl status.

The equivalent direct request is:

```bash
curl -sS http://127.0.0.1:19180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4",
    "messages": [{"role": "user", "content": "Explain JetsonFabric."}],
    "max_tokens": 16
  }' | jq
```

The client does not select stage count or co-location. The one-node development
cluster defaults to one pipeline stage.

To reuse already-built binaries:

```bash
JF_SKIP_BUILD=true make dev-up
```

## Stop or recover the dev processes

A foreground session can still be stopped with `Ctrl+C`.

From another terminal, use the scoped lifecycle command:

```bash
make dev-kill
```

`make kill` is an alias for the same target:

```bash
make kill
```

The command reads the recorded PID files, verifies that each process matches the
expected dev node or runtime command line, sends `SIGTERM`, waits for graceful
shutdown, and uses `SIGKILL` only if necessary. It does not broadly kill every
JetsonFabric process on the machine.

If PID files are missing or stale, the command searches only for processes using
the fixed dev ports and the `jf-dev` identity. If either port remains occupied by
an untracked process, it reports the listener rather than killing it blindly.

## Unit tests

Run Go unit tests:

```bash
make test
```

Native C++ tests are run by CI after the CPU runtime build. They include both:

- one-stage full-model prefill/decode equivalence;
- two-stage activation-handoff prefill/decode equivalence.

## Real-model integration tests

Run a deterministic one-stage CPU integration test:

```bash
make test-integration-single \
  MODEL_PATH=models/qwen.gguf \
  MODEL=qwen2.5-coder-1.5b-q4
```

Run the explicit two-stage co-located CPU integration test:

```bash
make test-integration-pipeline \
  MODEL_PATH=models/qwen.gguf \
  MODEL=qwen2.5-coder-1.5b-q4
```

The pipeline integration harness is not a normal serving mode. It exists to
prove stage ordering, real activation transport, persistent decode state, token
equivalence, and session cleanup on one machine.

CI uses the small `stories15M-q4_0.gguf` model with short prompts for both real-
model integration paths. A separate synthetic integration test validates binary
stage transport without loading a model.

## Current limitation

Stage-local tensor loading, automatic placement reconciliation, and safe epoch
handoff are implemented for Llama and Qwen2. Rebalance temporarily needs memory
for old and new partitions on reused nodes. Production-style multi-stage
validation still requires distinct physical Jetsons; co-location proves
correctness and memory partition contracts, not distributed CUDA performance.
