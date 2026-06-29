# Architecture

See [Architecture Diagrams](architecture-diagrams.md) for component, sequence,
contract, and node lifecycle diagrams.

JetsonFabric has four layers:

1. Control plane
2. Node agents
3. Runtime workers
4. Benchmark and routing policy

## Control Plane

The control plane is Go-native.

The control plane owns the cluster model:

- nodes
- models
- deployments
- placement previews
- metrics
- benchmark history
- routing decisions

It exposes a local API and, later, a dashboard plus an OpenAI-compatible API.

## Node Agents

Node agents are Go-native so they can run as a small systemd-managed binary on
Jetson devices.

Each Jetson runs an agent that sends heartbeats and capability data to the
control plane. A new Jetson joins the cluster by installing this agent and
providing the control-plane URL plus a join token.

Agents also expose the node-facing model proxy API. The control plane should
route model requests to the agent, and the agent should forward them to the
node-local runtime such as `llama-server`, TensorRT, Triton, or a future custom
C++ worker. This keeps runtime ports, model paths, and node-local process details
out of the control plane.

The agent should eventually collect:

- Jetson model
- JetPack version
- CUDA/TensorRT availability
- memory
- temperature
- power mode
- throttling state
- loaded model shards
- runtime health

For P0, the agent knows:

- which local runtime URL to proxy to
- which JetsonFabric model IDs that runtime serves
- which agent URL should be advertised back to the control plane

Model downloads and runtime launch can remain script-managed until the first
single-Jetson demo works. A later model artifact manifest should move model
source URLs, local paths, expected hashes, and launch commands under explicit
agent/runtime management.

## Runtime Workers

Runtime workers are adapters around actual model engines:

- C++ llama.cpp adapter for quantized LLM baselines
- C++ TensorRT/ONNX adapter for optimized inference
- Triton adapter for production-style serving experiments
- custom C++ layer-split runtime for distributed transformer inference

Python is only for benchmark analysis, graphing, and offline report generation.

Runtime workers should be C++ first. C is acceptable only at external API
boundaries such as POSIX sockets, CUDA, `libibverbs`, or vendor runtime
libraries. Wrap those C APIs in small C++ types so tensor buffers, sessions,
sockets, CUDA handles, and registered memory have clear ownership.

CUDA is not a default dependency for every service. It enters the runtime lane
when JetsonFabric needs GPU-aware behavior:

- TensorRT or llama.cpp GPU integration
- pinned or mapped host buffers
- CPU/GPU transfer measurement
- activation compression kernels
- possible GPUDirect/RDMA experiments after TCP baselines exist

The Go control plane should never own raw tensor bytes. It selects routes,
records benchmarks, and instructs runtime workers. The C++ runtime owns
layer-shard execution and tensor transport.

## Tensor Transport

Layer-split runtime work should start with a simple TCP transport. The first
wire protocol should be explicit and boring:

- request or session ID
- decode step
- source and target layer boundary
- batch size
- sequence length
- hidden dimension
- dtype
- byte length
- payload bytes

Use typed enums in code for fields such as dtype and route/runtime mode, with
stable numeric values on the wire. Do not serialize raw C++ structs directly;
encode and decode fields explicitly so padding, alignment, and endian behavior
do not become hidden protocol contracts.

Transport phases:

1. Built-in network with TCP for the P1 layer-split baseline.
2. Optional 10GbE TCP using the same JetsonFabric tensor protocol.
3. Optional RDMA transport only after measurements show TCP or CPU copy overhead
   is the bottleneck.
4. Tensor-parallel experiments only after layer split and transport data justify
   the synchronization cost.

## Placement Modes

V0 supports route previews only. Planned modes:

- single-node
- replica_serving
- layer split
- tensor-parallel experiment
- Codex/cloud fallback

Replica mode is a baseline, not the product identity. Layer split and
profile-driven placement are the main research direction.
