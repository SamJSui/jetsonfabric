# Local development

The normal developer path runs one Go node supervising one C++ runtime. That
runtime is stage `0/1` and executes the complete model layer range through the
same pipeline protocol used by multi-node deployments.

Co-located multi-stage execution is test-only. It is intentionally kept out of
`make dev-up` because every current test runtime still loads the complete GGUF,
which creates artificial memory pressure on small Jetsons.

## Configuration

Copy the template and point it at a local GGUF:

```bash
cp .env.example .env.local
```

A Jetson Orin development configuration can use:

```dotenv
MODEL=qwen2.5-coder-1.5b-q4
MODEL_PATH=models/qwen.gguf
RUNTIME_CUDA_ARCH=87
RUNTIME_COMPUTE_BACKEND=cuda
RUNTIME_MODE=pipeline_parallel
RUNTIME_CTX_SIZE=4096
RUNTIME_N_GPU_LAYERS=999
```

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

1. loads `.env.local` through the Makefile;
2. builds the selected CUDA or CPU runtime when needed;
3. builds a native `jetsonfabric-node`;
4. reads the model's actual transformer-layer count;
5. starts one Go node and one supervised C++ runtime;
6. configures one full-range pipeline stage;
7. waits for health;
8. sends a real one-token chat request;
9. prints ready only after inference succeeds;
10. remains in the foreground until `Ctrl+C`.

The default node URL is:

```text
http://127.0.0.1:19180
```

Logs are retained while the node is running under:

```text
.cache/jetsonfabric/dev/logs/
```

## Inspect and use the service

In another terminal:

```bash
make dev-status
```

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

Stop the node with `Ctrl+C`. The launcher terminates the Go node, which then stops
its supervised C++ runtime.

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

Partial-layer graph execution is implemented, but each runtime still loads the
complete GGUF. True stage-local tensor loading and dynamic deployment rebalancing
belong to later milestones. Until then, production-style multi-stage validation
should use distinct physical Jetsons; co-location remains a minimal CPU test.
