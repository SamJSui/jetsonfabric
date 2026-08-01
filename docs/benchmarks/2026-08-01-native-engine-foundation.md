# Native Engine Foundation: 14B Stage Selection And Readiness

This experiment validates the first JetsonFabric-native engine boundary: GGUF
import into a cluster-size-independent JFM v2 package, exact layer-stage mapping,
segment integrity, and NVMe-to-memory first-touch behavior. It does **not**
measure native token generation; llama.cpp remains the serving engine.

## Configuration

| Item | Value |
| --- | --- |
| Node | `dopey`, Jetson Orin Nano 8GB Super |
| Jetson Linux | R39.2, kernel 6.8.12-1021-tegra |
| Power mode | MAXN_SUPER, clocks fixed with `jetson_clocks` |
| Storage | SPCC M.2 PCIe NVMe SSD |
| Model | Qwen2.5-Coder-14B-Instruct Q4_K_M |
| Source GGUF | 8,988,110,272 bytes |
| Source SHA-256 | `c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940` |
| JFM package | v2, 8,988,179,784 bytes |
| JFM tensors | 579 tensors, 48 layers |
| Code provenance | `feature/tensor-parallel-runtime`, base `3626ce7`, dirty worktree |

Raw records and machine provenance are under
[`evidence/20260801-native-engine`](evidence/20260801-native-engine/).

## Exact Stage Selection

| Stage | Layers | Selected weights | Share | Tensors |
| --- | ---: | ---: | ---: | ---: |
| First | `[0, 24)` | 4,390,699,008 bytes | 48.88% | 289 |
| Last | `[24, 48)` | 4,591,443,968 bytes | 51.12% | 290 |
| Full package | `[0, 48)` | 8,982,142,976 bytes | 100% | 579 |

The first stage owns the input embedding. The last stage owns output norm and
projection, which explains the asymmetry. Changing the split only changes the
opened segment set; it does not rebuild the package.

## Warm Mapping

Mapping validates the manifest and tensor indexes without claiming every weight
page is already resident in RAM.

| Range | Iterations | Median open | p95 open |
| --- | ---: | ---: | ---: |
| `[0, 24)` | 20 | 0.356 ms | 0.396 ms |
| `[24, 48)` | 20 | 0.353 ms | 0.380 ms |
| `[0, 48)` | 20 | 0.685 ms | 0.717 ms |

This is metadata latency, not readiness latency. The CUDA execution path must
still fault or copy selected weights before inference.

## Cold First-Touch Optimization

The benchmark asks Linux to evict clean segment pages with `posix_fadvise`,
maps a stage, then reads one byte from every page. Early JFM v1 tuning over the
4.39 GB first stage identified the access policy and thread count:

| Access hint | Threads | Median | Effective rate |
| --- | ---: | ---: | ---: |
| Random | 1 | 45.181 s | 0.10 GB/s |
| Sequential | 1 | 2.881 s | 1.52 GB/s |
| Sequential | 2 | 2.092 s | 2.10 GB/s |
| Sequential | 4 | 1.989 s | 2.21 GB/s |
| Sequential | 6 | 1.954 s | 2.25 GB/s |

Correcting `MADV_RANDOM` to `MADV_SEQUENTIAL` reduced one-thread first-touch by
93.6%. Four threads produced a 22.7x improvement over the original path and
was within 1.8% of six threads, so four is the default. The second 4.59 GB stage
reached first-touch in 2.243 s with four threads, or 2.05 GB/s.

The hardened v2 loader then measured five cold approximations per stage. It
includes the preserved 6 MB GGUF metadata mapping and reports the nearest-rank
p95 rather than interpolating small samples.

| Stage | Median prefetch | Median ready | Maximum ready |
| --- | ---: | ---: | ---: |
| `[0, 24)` | 2.178 s | 2.649 s | 2.697 s |
| `[24, 48)` | 2.205 s | 2.700 s | 2.735 s |

These are non-root cold-cache approximations. `POSIX_FADV_DONTNEED` is an OS
hint and is not identical to a reboot or privileged global cache drop.

## Import Optimization

The importer preserves the original quantized tensor bytes and computes the
source plus per-segment SHA-256 values.

| SHA implementation | Import time |
| --- | ---: |
| Portable scalar, separate segment pass | 243 s |
| Portable scalar, hash during writes | 238 s |
| Dynamic OpenSSL acceleration, hash during writes | 41 s |
| JFM v2, OpenSSL plus source re-verification | 55.2 s |

Using the device's OpenSSL implementation reduced prototype import time by
83.1% while retaining the portable implementation as fallback. JFM v2 spends
an additional source-hash pass to reject concurrent input changes. Full v2
package verification with the dependency-free scalar reader took 108.1 s and is therefore a
provisioning/integrity gate, not a per-deployment hot-path operation.

## Conclusion

With five samples, the last column is the observed maximum rather than a stable
tail-latency estimate.

The foundation proves a useful ownership boundary: each runtime can select only
its assigned 14B weights from one reusable artifact, and optimized NVMe
first-touch takes about 2.2 seconds per half on this device. It does not yet
prove faster inference. The next gate is one-node Qwen correctness through a
native executor, followed by CUDA kernel and end-to-end prefill/decode
measurements against llama.cpp.
