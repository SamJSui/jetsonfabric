# Physical two-Jetson CUDA validation

This is the final hardware gate for topologically distributed pipeline parallelism.
It must run against two distinct Jetson hosts after the CPU and synthetic CI gates
are green.

## Required deployment

Each Jetson runs one `jetsonfabric-node` supervising one CUDA-enabled
`jetsonfabric-runtime-worker` built from the same commit and the same pinned
llama.cpp revision.

The logical nodes must advertise:

- different physical hostnames;
- `compute_backends` containing `cuda`;
- the same model ID and model artifact;
- `pipeline_parallel` mode;
- contiguous, non-overlapping layer ranges;
- one shared stage count.

For a two-stage model with `N` transformer layers:

```text
Jetson A: stage_index=0, layer range [0, split)
Jetson B: stage_index=1, layer range [split, N)
```

Do not set `allow_colocated_stages=true` for the hardware acceptance run.

## Build requirement

Build both runtimes with CUDA enabled and apply the pinned JetsonFabric llama.cpp
stage-range patch through the normal CMake configuration:

```bash
cmake -S runtime -B runtime/build-jetson \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build runtime/build-jetson --parallel
```

The build fails if the pinned stage-range extension is absent.

## Validation command

Run the harness from a machine that can reach the coordinator node API:

```bash
JF_COORDINATOR_URL=http://jetson-a:8080 \
JF_MODEL_ID=qwen2.5-coder-1.5b-q4 \
JF_STAGE_COUNT=2 \
JF_MAX_TOKENS=2 \
bash scripts/jetson/validate-distributed-cuda.sh
```

To compare against a separately measured one-runtime greedy baseline, pass the
expected token IDs as JSON:

```bash
JF_EXPECTED_TOKENS='[1234,5678]' \
JF_COORDINATOR_URL=http://jetson-a:8080 \
JF_MODEL_ID=qwen2.5-coder-1.5b-q4 \
bash scripts/jetson/validate-distributed-cuda.sh
```

## Acceptance criteria

The harness requires:

- at least two distinct CUDA-capable physical hosts in membership;
- a valid `pipeline_parallel` route with `topology=distributed`;
- at least two physical hosts in the selected plan;
- real activation payloads between stages;
- activation byte-count and CRC continuity;
- a sampled token from the final stage;
- decode-step traces when more than one token is produced;
- optional exact greedy-token equality with a one-runtime baseline.

Record the harness output together with `tegrastats` from both Jetsons. The final
P0 evidence should include per-stage latency, activation bytes, GPU utilization,
memory use, temperature, and total tokens per second.

## What CPU CI proves first

CPU CI is responsible for proving:

- the pinned llama.cpp patch applies and compiles;
- split prefill produces a real hidden state;
- a downstream runtime resumes from that hidden state;
- greedy first-token and decode-token equivalence;
- stage contexts persist by session ID;
- the coordinator drives decode and closes every session;
- activation bytes cross two real logical-node APIs.

The physical harness adds topology and CUDA evidence. It is not considered passed
until run on the actual Jetsons.
