# JetsonFabric Native Engine

The native engine isolates the model hot path from node orchestration and from
`libllama`. It includes an offline GGUF importer, a dependency-light C loader,
and an architecture-selected inference path over GGML backends.

The `native` serving engine implements Qwen2/Qwen2.5 attention, FFN, logits,
optional output bias, greedy token selection, and incremental F16 KV caching.
It currently serves a full model on one node. llama.cpp remains the default
engine and correctness oracle while native partial-layer execution is built.

## Boundaries

```text
jetsonfabric-node (Go)
  -> jetsonfabric-runtime-worker (C++)
      -> inference::Executor
          -> llama.cpp executor (current serving path)
          -> native executor (full-model serving)
              -> NativeEngine
                  -> TensorStore (JFM integrity, tensors, GGML backend)
                  -> ArchitectureRegistry
                      -> Qwen2Architecture (first strategy)
```

`NativeEngine` is not Qwen-specific. Model architecture modules own only the
metadata rules, required tensor shapes, and compute graph that differ between
model families. Adding another family means implementing `ModelArchitecture`
and registering its GGUF architecture name; it does not change JFM loading,
backend selection, generation control, or benchmark reporting.

The C loader does not depend on Go, HTTP, JSON, discovery, coordinator state,
GGML, or llama.cpp. The offline compiler uses the pinned GGUF parser only to
import metadata and copy the original quantized tensor bytes. Native inference
uses JetsonFabric-owned graphs over GGML/GGML-CUDA and does not call
`libllama`. The serving adapter temporarily uses llama.cpp's vocabulary code to
tokenize preserved `metadata.gguf`; it does not load a second copy of weights.

## Package Format

`jf-model-compile` converts one GGUF into a reusable JFM directory:

```text
model.jfm/
  manifest.jfm
  metadata.gguf
  shared.jfs
  input.jfs
  layer-00000.jfs
  layer-00001.jfs
  ...
  output.jfs
```

The package is independent of cluster size. A two-stage deployment opens
layers `[0, N)` on one node and `[N, layer_count)` on another without creating
new model artifacts. The first stage also maps input tensors; the last stage
maps output tensors; shared tensors are available to every stage.

JFM v2 preserves the complete GGUF metadata and tensor directory, then records
the source SHA-256, layer count, stable JetsonFabric tensor types, dimensions,
quantization block geometry, segment sizes, and per-segment SHA-256 values.
Opening with hash verification detects corruption before execution. Tensor
payload bytes are preserved exactly, including GGUF quantization. The loader
rejects missing or duplicate layers, paths, and selected tensor names as well
as inconsistent shapes, block sizes, byte totals, and overlapping payloads.
See [JFM v2 binary format](jfm-v2-format.md) for the compatibility contract.

Hashes detect corruption, not authenticity. Operators must compare the source
hash exposed by the loader with trusted registry metadata before accepting a
package copied from another host.

## Build And Import

Build the standalone hot library and its tests without llama.cpp:

```bash
make native-engine
```

Build the offline importer and benchmark, then import a model:

```bash
make model-compile \
  MODEL_PATH=/var/lib/jetsonfabric/models/model.gguf \
  JFM_PACKAGE=/var/lib/jetsonfabric/models/model.jfm
```

The destination must not already exist. The importer locks one source file
descriptor, verifies its SHA-256 before and after conversion, fsyncs an
exclusive staging directory, and atomically publishes the completed package.
An interrupted import cannot expose a partial destination.

## Native Serving

Select the native engine with `NODE_ENGINE=native` and pass the compiled JFM
directory as `MODEL_PATH`. M2 requires one stage containing the complete layer
range:

```bash
make run-node \
  NODE_ENGINE=native \
  MODEL=qwen2.5-coder-1.5b-q4 \
  MODEL_PATH=/var/lib/jetsonfabric/models/model.jfm \
  STAGE_INDEX=0 \
  STAGE_COUNT=1 \
  LAYER_START=0 \
  LAYER_END=28
```

The node reads the source GGUF SHA-256 from `manifest.jfm`, so registry and
deployment identity remains the same before and after compilation. To compile
a temporary package and verify real text generation against llama.cpp greedy
tokens through the node API, run:

```bash
MODEL_PATH=/var/lib/jetsonfabric/models/model.gguf \
JF_ENGINE=native \
bash scripts/local/validate-single-node.sh
```

## Loader Benchmark

Measure stage mapping plus first-touch page residency separately:

```bash
make bench-native-model \
  JFM_PACKAGE=/var/lib/jetsonfabric/models/model.jfm \
  JFM_LAYER_START=0 \
  JFM_LAYER_END=14 \
  JFM_BENCH_ITERATIONS=5 \
  JFM_BENCH_VERIFY=false \
  JFM_BENCH_EVICT=false \
  JFM_PREFETCH_THREADS=4 \
  JFM_BENCH_PREFETCH=true
```

`open` measures manifest validation, file mapping, and tensor-index creation.
`prefetch` measures faulting the selected mapped pages into memory; `ready`
measures both phases. Hash verification and prefetch must be benchmarked
separately because verification itself faults every page. Repeated
iterations are warm page-cache measurements unless the operator explicitly
drops the Linux page cache between runs. `JFM_BENCH_EVICT=true` asks Linux to
evict each clean segment before mapping it, providing a non-root approximation
of cold NVMe residency. `JFM_PREFETCH_THREADS` measures whether concurrent
segment faulting improves that storage-bound phase on the target device.

## Native Inference Benchmark

The inference benchmark consumes explicit token IDs so tokenizer behavior is
not mixed with graph correctness. Use the separate llama.cpp oracle to obtain
the prompt and expected greedy token sequences, then require an exact match:

```bash
dist/jf-llama-greedy-oracle \
  --model /var/lib/jetsonfabric/models/model.gguf \
  --prompt 'Once upon a time' \
  --max-tokens 4 \
  --n-gpu-layers 999 \
  --threads 6

make bench-native-inference \
  JFM_PACKAGE=/var/lib/jetsonfabric/models/model.jfm \
  JFM_INFERENCE_BACKEND=cuda \
  JFM_TOKENS=token0,token1,token2 \
  JFM_MAX_TOKENS=4 \
  JFM_EXPECTED_TOKENS=next0,next1,next2,next3 \
  JFM_INFERENCE_WARMUPS=1 \
  JFM_INFERENCE_ITERATIONS=10
```

The benchmark verifies JFM segment hashes, reports the source GGUF SHA-256 and
actual GGML backend/device, and emits raw timing samples. Incremental decode is
the default: prefill populates an F16 KV cache, then each decode step processes
only the sampled token. The output declares `kv_cache: true` and
`decode_policy: incremental_kv_cache`.

On supported CUDA builds, automatic attention selects Flash for prefill and
decode. The Qwen2 session preserves its logical request capacity while aligning
the physical Flash KV cache to GGML's 256-token fast-path stride. Explicit
`unfused` and `flash` options remain available for correctness and A/B tests.

Use `JFM_DECODE_POLICY=full-prefix` to rerun the historical full-prefix policy
as an ablation. That mode declares `kv_cache: false` and
`decode_policy: full_prefix_recompute`; it is not representative of llama.cpp,
which also uses incremental KV-cached decode.

Run the matched native-versus-llama.cpp workload matrix with explicit prompt
token IDs:

```bash
python3 tools/bench/native_engine_matrix.py \
  --native-bin dist/jf-native-inference-bench \
  --native-parity-bin dist/jf-native-attention-parity \
  --llama-bin dist/jf-llama-greedy-oracle \
  --package /var/lib/jetsonfabric/models/model.jfm \
  --model /var/lib/jetsonfabric/models/model.gguf \
  --prompt-lengths 32,128,512,2048 \
  --output-lengths 32,128,512 \
  --warmups 1 \
  --iterations 20 \
  --revision "$(git rev-parse --short HEAD)" \
  --output native-engine-matrix.json
```

The runner alternates engine order, retains raw timing vectors and executable
hashes, compares full-vocabulary logits before timing, and fails on any model,
prompt-token, or generated-token mismatch.

## Native Serving Gates

The first three gates enabled the current full-model serving path. Remaining
gates control the move to distributed native execution:

1. Greedy token IDs match llama.cpp for fixed real-model prompts on CPU and CUDA.
2. Logits stay within a documented tolerance.
3. Prefill, decode, and KV-cache rollback pass lifecycle tests.
4. One-node CUDA generation is stable before distributed stage execution.
5. Physical two-node tests prove exact residency and activation continuity.
6. Benchmarks include branch, commit, model hash, power mode, workload, and raw
   evidence before public performance claims are updated.
