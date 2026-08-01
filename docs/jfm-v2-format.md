# JFM v2 Binary Format

JFM is JetsonFabric's little-endian, directory-based model package. Version 2
preserves GGUF metadata while grouping tensor payloads by execution ownership.
Readers must reject unknown versions, reserved-field changes, incomplete model
topologies, and values outside the bounds documented here.

## Package Topology

A package contains exactly one `manifest.jfm`, `metadata.gguf`, input segment,
output segment, and segment for every layer. One optional shared segment is
allowed. Segment paths are single safe file names and must be unique.

`metadata.gguf` is the exact source byte range `[0, gguf_data_offset)`. It
contains the GGUF header, key-value metadata, and tensor directory but no
tensor payloads.

## Manifest

The 64-byte manifest header is:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `JFMODEL2` |
| 8 | 4 | format version, `2` |
| 12 | 4 | transformer layer count |
| 16 | 8 | source GGUF byte size |
| 24 | 32 | source GGUF SHA-256 |
| 56 | 4 | segment count |
| 60 | 4 | reserved, zero |

Each segment record is a 72-byte fixed prefix followed immediately by
`path_length` bytes with no terminator or padding:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 4 | kind: shared `0`, input `1`, layer `2`, output `3`, metadata `4` |
| 4 | 4 | signed layer index; `-1` for non-layer segments |
| 8 | 8 | tensor count; zero only for metadata |
| 16 | 8 | sum of tensor payload bytes; zero only for metadata |
| 24 | 8 | segment file size |
| 32 | 32 | SHA-256 of the complete segment file |
| 64 | 4 | path length, 1 through 255 |
| 68 | 4 | reserved, zero |

## Tensor Segments

Tensor segment files begin with a 64-byte header:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | ASCII `JFSEGM02` |
| 8 | 4 | format version, `2` |
| 12 | 4 | tensor count |
| 16 | 8 | tensor-index bytes after the header |
| 24 | 8 | sum of tensor payload bytes |
| 32 | 32 | reserved, zero |

Each variable index entry contains an 80-byte record, `name_length` name bytes,
then zero padding to eight-byte alignment:

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 4 | stable JFM tensor type |
| 4 | 4 | rank, 1 through 4 |
| 8 | 4 | signed layer index |
| 12 | 4 | flags, zero |
| 16 | 32 | four `uint64` dimensions; unused dimensions are one |
| 48 | 8 | payload offset in this segment |
| 56 | 8 | payload byte length |
| 64 | 4 | tensor-name length, 1 through 1024 |
| 68 | 4 | canonical elements per storage block |
| 72 | 4 | canonical bytes per storage block |
| 76 | 4 | reserved, zero |

Payload data begins at `align64(64 + index_bytes)`. Every payload offset is
64-byte aligned, payload ranges are ordered and non-overlapping, and the file
is padded to 64-byte alignment. Quantization blocks apply along dimension zero;
`shape[0]` must be divisible by the canonical block element count.

## Stable Tensor Types

| IDs | Types | Block elements / bytes |
| --- | --- | --- |
| 1-2 | F32, F16 | `1/4`, `1/2` |
| 3-8 | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | `32/18`, `32/20`, `32/22`, `32/24`, `32/34`, `32/36` |
| 9-14 | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | `256/84`, `256/110`, `256/144`, `256/176`, `256/210`, `256/292` |
| 15-22 | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS | `256/66`, `256/74`, `256/98`, `256/50`, `32/18`, `256/110`, `256/82`, `256/136` |
| 23-29 | I8, I16, I32, I64, F64, IQ1_M, BF16 | `1/1`, `1/2`, `1/4`, `1/8`, `1/8`, `256/56`, `1/2` |
| 30-35 | TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 | `256/54`, `256/66`, `32/17`, `64/36`, `128/18`, `64/18` |

The dependency-light loader owns this table. The GGUF importer refuses to
publish a package if the pinned GGML revision reports different geometry.

## Integrity And Trust

Segment and source hashes detect accidental corruption. They do not establish
authenticity because the manifest itself is unsigned. A provisioning system
must compare the source hash to trusted registry metadata, or authenticate the
entire manifest with a signature, before accepting a package from another
machine.

The committed hex package under
`runtime/engines/native/tests/fixtures/jfm-v2-golden` freezes the byte contract
independently of the current writer. Any incompatible change requires a new
format version rather than silently changing v2.
