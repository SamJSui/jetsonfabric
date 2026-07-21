# Single-Node Runtime Validation

The single-node path validates the same facade-to-runtime contract used by a
multi-stage deployment, with one node assigned the full model layer range:

```text
client -> jetsonfabric-node -> jetsonfabric-runtime-worker -> llama.cpp
```

## Build

```bash
make node
make runtime RUNTIME_BUILD_JOBS=2
```

On an Orin, use `make runtime-cuda RUNTIME_BUILD_JOBS=1` first. Increase build
parallelism only after observing available memory.

## Real-Model Test

```bash
make test-integration-single \
  MODEL=qwen2.5-coder-1.5b-q4 \
  MODEL_PATH=/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

The harness starts one `jetsonfabric-node`, lets it supervise one runtime,
executes prefill and repeated decode through the public node API, and requires
exact greedy-token equality with the native full-model baseline.

## Lifecycle Test

```bash
MODEL_PATH=/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
MODEL_ID=qwen2.5-coder-1.5b-q4 \
bash scripts/local/validate-runtime-lifecycle.sh
```

This covers `idle -> load -> ready -> activate -> infer -> unload -> idle` and
checks deployment identity and model-residency metadata at every barrier.

Passing these tests proves node supervision, lifecycle, and real one-stage
inference. It does not prove multi-host execution or aggregate cluster memory.
