# Project Context

JetsonMesh is a Jetson-first edge AI cluster project. It borrows the strongest
ideas from exo while narrowing the market and engineering story to cheaper,
smaller edge devices.

## One-Sentence Goal

Build an exo-inspired distributed inference runtime for low-cost Jetson edge
clusters that can prove, with benchmarks, when orchestrating multiple small
devices improves serving cost, reliability, latency, throughput, memory fit, or
deployment flexibility.

## Why Build This

Most AI infrastructure work is about orchestrating model work across compute
rather than simply running a model on one machine. JetsonMesh turns a mini edge
cluster into a concrete distributed systems project:

- node discovery and registration
- hardware capability profiling
- model registry and compatibility checks
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

JetsonMesh should mimic that user experience:

- start control plane
- install/run agents on devices
- see nodes appear
- see model compatibility and route previews
- send requests through one API
- inspect why the system chose a route

The difference is focus. exo commonly showcases stronger consumer machines such
as Macs. JetsonMesh focuses on edge-class ARM/GPU devices where memory, thermals,
power, and network transfer are central constraints.

## Performance Story

The project must earn any performance claim through benchmarks. Compare:

- single-node Jetson baseline
- replicated serving baseline
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

## Layer Split Versus Tensor Parallel

Layer split assigns different model layers to different nodes. A request flows
through node A, then node B, and so on. It is easier to reason about on ordinary
Ethernet because communication happens at layer boundaries.

Tensor parallel splits individual matrix operations across nodes. It can be
powerful on fast interconnects, but it usually requires frequent synchronization.
On Jetson devices connected by normal Ethernet, network transfer and
synchronization can erase the compute benefit. JetsonMesh can research it later,
but the first serious distributed-runtime milestone should be layer split.

## Hardware Strategy

Start with one Jetson Orin Nano-class device to establish the runtime,
benchmarks, and model backend. Add a second Jetson once the baseline is real.

Recommended path:

1. Beelink or dev machine runs the control plane.
2. One Jetson runs the agent and a small model backend.
3. Record baseline model performance and thermal behavior.
4. Add a second Jetson and benchmark replica/failover.
5. Prototype layer-split execution for a small transformer.

Raspberry Pi nodes are not the core inference performance story. They may become
useful later for sensors, health probes, gateway tests, or extremely low-power
edge roles.

## Implementation Shape

V0 services:

- `jetsonmesh-control`: Go API gateway, node registry, model registry, route
  preview, and future scheduler.
- `jetsonmesh-agent`: Go binary installed on each node to report capabilities,
  health, runtimes, and benchmark results.
- runtime adapters: C++ integrations for llama.cpp, TensorRT/ONNX, Triton, and
  eventually custom layer-shard execution.
- benchmark/reporting lane: Python scripts or notebooks only for analysis and
  visualization.

## Ideal State

A user can connect a new Jetson, install the agent, provide the control-plane
URL and join token, and watch it become eligible for model work after calibration
benchmarks. The system then serves through one API, explains its placement
decision, records performance evidence, and can show when edge-local execution
is better or worse than another route.
