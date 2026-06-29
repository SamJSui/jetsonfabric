# P1 - Tensor Parallelism

P1 is a research milestone after layer split works. The goal is to test whether
splitting tensor operations across Jetson nodes can help enough to justify the
extra synchronization cost.

Tensor parallelism is harder than layer split because communication happens
inside layers, not only between layer ranges. On ordinary Ethernet this can lose
quickly. The milestone is still useful if it produces a well-measured negative
result.

## Goal

Prototype a narrow tensor-parallel path for one model component, such as an
attention projection or MLP projection, and compare it against:

- single-node POC serving;
- P0/MVP layer split;
- replica serving under concurrent load.

## Required Measurements

- synchronization frequency;
- network bytes/token;
- per-step latency;
- GPU utilization;
- CPU copy overhead;
- quality or numerical drift if reduced precision is used;
- whether the result improves memory fit, latency, throughput, or nothing.

## Acceptance Criteria

P1 is complete when the project can answer:

```text
Does tensor parallelism make sense on this Jetson network for this model class?
```

The answer can be "no" if measurements show synchronization dominates.
