# Local development

The current public chat path requires a pipeline with at least two logical stages.
A one-stage `.env.local` is still useful for starting and inspecting one supervised
runtime, but it does not exercise distributed `/v1/chat/completions`.

## Configuration

Copy the current template and point it at a local GGUF:

```bash
cp .env.example .env.local
```

The Qwen2.5 Coder 1.5B Q4 development model should use:

```dotenv
MODEL=qwen2.5-coder-1.5b-q4
MODEL_PATH=models/qwen.gguf
RUNTIME_CUDA_ARCH=87
RUNTIME_COMPUTE_BACKEND=cuda
RUNTIME_MODE=pipeline_parallel
RUNTIME_CTX_SIZE=4096
RUNTIME_N_GPU_LAYERS=999
```

`RUNTIME_BUILD_JOBS=4` is reasonable if the Jetson builds reliably. Reduce it to
`2` or `1` if compilation is killed or the system becomes memory constrained.

The base stage values may remain:

```dotenv
STAGE_INDEX=0
STAGE_COUNT=1
LAYER_START=0
LAYER_END=28
```

`make dev-up` intentionally ignores those four values. It reads the GGUF's actual
layer count and starts two complementary logical nodes on the same machine.

## Build and run a persistent colocated cluster

From the repository root:

```bash
make test
make dev-up
```

`make dev-up`:

1. loads `.env.local` through the Makefile;
2. builds the CUDA or CPU runtime selected by `RUNTIME_COMPUTE_BACKEND`;
3. builds a native `jetsonfabric-node` for the current machine;
4. reads the model's actual layer count;
5. starts two Go nodes and two supervised C++ runtimes;
6. waits for both nodes and a valid colocated route;
7. remains in the foreground until Ctrl+C.

Default node URLs are:

```text
http://127.0.0.1:19180
http://127.0.0.1:19181
```

Logs are retained while the cluster is running under:

```text
.cache/jetsonfabric/colocated-dev/logs/
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

The equivalent direct request is:

```bash
curl -fsS http://127.0.0.1:19180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4",
    "messages": [{"role": "user", "content": "Explain JetsonFabric."}],
    "max_tokens": 16,
    "jetsonfabric": {
      "stage_count": 2,
      "allow_colocated_stages": true
    }
  }' | jq
```

To reuse already-built binaries:

```bash
JF_SKIP_BUILD=true make dev-up
```

Stop the cluster with Ctrl+C. The launcher terminates both Go nodes; each node
then stops its supervised C++ runtime.

## Focused validation

The persistent launcher is for exploration. The existing Phase 1 script performs
token-equivalence and activation-continuity assertions and exits when complete:

```bash
MODEL_PATH=models/qwen.gguf \
MODEL_ID=qwen2.5-coder-1.5b-q4 \
bash scripts/local/validate-colocated-pipeline.sh
```

That validation currently builds and runs the CPU path for deterministic CI-style
correctness evidence. Rebuild with `make runtime-cuda` afterward when returning to
interactive CUDA development.
