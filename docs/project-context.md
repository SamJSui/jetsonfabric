# Project Context

JetsonFabric is a Jetson-first edge AI cluster project. It borrows the strongest
ideas from exo while narrowing the market and engineering story to cheaper,
smaller edge devices.

## One-Sentence Goal

Build an exo-inspired distributed inference runtime for low-cost Jetson edge
clusters that can prove, with benchmarks, when orchestrating multiple small
devices improves serving cost, reliability, latency, throughput, memory fit, or
deployment flexibility.

## Why Build This

Most AI infrastructure work is about orchestrating model work across compute
rather than simply running a model on one machine. JetsonFabric turns a mini edge
cluster into a concrete distributed systems project:

- node discovery and registration
- hardware capability profiling
- model registry and compatibility checks
- agent-owned model artifact and runtime metadata
- scheduler and placement decisions
- route explanations
- benchmark-driven policy
- failure handling and degraded routing
- later OpenAI-compatible API serving

The value-add is not "I can run Qwen locally." The value-add is:

- I can explain what work belongs on which edge node and why.
- I can measure when a second node helps and when network transfer hurts.
- I can show cost/performance tradeoffs for edge deployments.
- I can make distributed inference observable instead of hand-wavy.
- I can add nodes by installing an agent rather than manually rewriting the
  deployment.

## Target Roles

This project should appeal to AI infrastructure, ML systems, edge AI, robotics
platform, and distributed systems roles. The strongest interview signals are:

- Go-native control plane and agent design
- scheduler and route planning
- observability across heterogeneous devices
- benchmark methodology
- C++ runtime integration for latency-sensitive inference paths
- honest performance analysis
- edge constraints: memory, thermals, power, bandwidth, and reliability

## Similarity To exo

exo makes distributed local AI feel automatic: discover devices, understand
their resources, choose a placement strategy, and expose a simple UX/API.

JetsonFabric should mimic that user experience:

- start control plane
- install/run agents on devices
- see nodes appear
- see model compatibility and route previews
- send requests through one API
- inspect why the system chose a route

The difference is focus. exo commonly showcases stronger consumer machines such
as Macs. JetsonFabric focuses on edge-class ARM/GPU devices where memory, thermals,
power, and network transfer are central constraints.

## Performance Story

The project must earn any performance claim through benchmarks. Compare:

- single-node Jetson baseline
- replica_serving
- layer-split distributed inference
- cloud/Codex/frontier fallback when local execution is not appropriate
- tensor-parallel experiments only if the network and runtime can support them

Expected metrics:

- tokens/sec
- p50/p95 latency
- time-to-first-token
- memory use
- temperature and throttling
- power mode
- network bytes/token
- failure and failover behavior
- quality or task pass rate for benchmark prompts

The first benchmark target is intentionally simple: one Jetson, one local model,
one prompt routed through JetsonFabric, and one recorded result. Distributed
runtime work starts only after that baseline exists.

## Phase Strategy

P0 is single-Jetson serving. The only goal is to make one real Jetson-backed
model work through the control plane and record trustworthy measurements.

P1 is multi-Jetson layer split. Replica serving belongs here only as a
comparison baseline for throughput and failover. The main question is whether
two or three Jetsons can run a larger or better model by assigning layer ranges
to different nodes and sending hidden states between workers.

P2 is distributed runtime optimization. This is where C++/CUDA transport work
belongs: activation framing, pinned buffers, compression, 10GbE TCP, optional
RDMA, and tensor-parallel experiments. P2 starts only after P1 measurements show
what bottleneck is worth optimizing.

## Layer Split Versus Tensor Parallel

Layer split assigns different model layers to different nodes. A request flows
through node A, then node B, and so on. It is easier to reason about on ordinary
Ethernet because communication happens at layer boundaries.

Tensor parallel splits individual matrix operations across nodes. It can be
powerful on fast interconnects, but it usually requires frequent synchronization.
On Jetson devices connected by normal Ethernet, network transfer and
synchronization can erase the compute benefit. JetsonFabric can research it later,
but the first serious distributed-runtime milestone should be layer split.

## Hardware Strategy

Start with one Jetson Orin Nano-class device to establish the runtime,
benchmarks, and model backend. Add a second Jetson once the baseline is real.

Recommended path:

1. Beelink or dev machine runs the control plane.
2. One Jetson runs the agent and a small model backend.
3. Record baseline model performance and thermal behavior.
4. Add a second Jetson and benchmark replica_serving/failover as a control comparison.
5. Prototype layer-split execution for a small transformer.
6. Add a third Jetson only after two-node measurements justify it.
7. Explore 10GbE or RDMA only after built-in-network measurements prove that
   transport is the limiting factor.

Raspberry Pi nodes are not the core inference performance story. They may become
useful later for sensors, health probes, gateway tests, or extremely low-power
edge roles.

## Implementation Shape

V0 services:

- `jetsonfabric-control`: Go API gateway, node registry, model registry, route
  preview, and future scheduler.
- `jetsonfabric-agent`: Go binary installed on each node to report capabilities,
  health, runtimes, model artifact metadata, benchmark results, and to proxy
  model requests to node-local runtimes.
- runtime adapters: C++ integrations for llama.cpp, TensorRT/ONNX, Triton, and
  eventually custom layer-shard execution.
- CUDA runtime work: pinned or mapped buffers, GPU transfer measurement,
  activation compression, TensorRT integration, and possible GPUDirect/RDMA
  experiments after TCP baselines exist.
- benchmark/reporting lane: Python scripts or notebooks only for analysis and
  visualization.

Current implementation priority:

1. Single-Jetson model backend adapter.
2. Control-plane routing to that backend.
3. Benchmark record for the routed response.
4. Node observability and route explanation.
5. Only then, second-node and layer-split work.

## Ideal State

A user can connect a new Jetson, install the agent, provide the control-plane
URL and join token, and watch it become eligible for model work after calibration
benchmarks. The system then serves through one API, explains its placement
decision, records performance evidence, and can show when edge-local execution
is better or worse than another route.
