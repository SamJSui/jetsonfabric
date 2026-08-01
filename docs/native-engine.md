# JetsonFabric Native Engine

The native engine work isolates the model hot path from node orchestration and
from llama.cpp. It currently implements the first production-facing boundary:
an offline GGUF importer and a dependency-light C library that maps only the
weights required by a deployment stage.

It is not yet registered as a serving engine. llama.cpp remains the inference
implementation and correctness oracle until the native path has real Qwen
tokenization, attention, KV-cache, logits, and sampling parity. This prevents
the repository from advertising a synthetic or incomplete inference path.

## Boundaries

```text
jetsonfabric-node (Go)
  -> jetsonfabric-runtime-worker (C++)
      -> inference::Executor
          -> llama.cpp executor (current serving path)
          -> native executor (after parity gates)
              -> libjetsonfabric-native-engine (C/CUDA)
```

The C library does not depend on Go, HTTP, JSON, discovery, coordinator state,
or llama.cpp. The offline compiler uses the pinned GGUF parser only to import
GGUF metadata and copy the original quantized tensor bytes.

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

## Native Serving Gates

The native implementation should be selectable through `inference::Executor`
only after all of these are true for one supported Qwen architecture:

1. Greedy token IDs match llama.cpp for fixed prompts.
2. Logits stay within a documented tolerance.
3. Prefill, decode, and KV-cache rollback pass lifecycle tests.
4. One-node CUDA generation is stable before distributed stage execution.
5. Physical two-node tests prove exact residency and activation continuity.
6. Benchmarks include branch, commit, model hash, power mode, workload, and raw
   evidence before public performance claims are updated.
