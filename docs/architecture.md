# Architecture

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

## Runtime Workers

Runtime workers are adapters around actual model engines:

- C++ llama.cpp adapter for quantized LLM baselines
- C++ TensorRT/ONNX adapter for optimized inference
- Triton adapter for production-style serving experiments
- custom C++ layer-split runtime for distributed transformer inference

Python is only for benchmark analysis, graphing, and offline report generation.

## Placement Modes

V0 supports route previews only. Planned modes:

- single-node
- replica baseline
- layer split
- tensor-parallel experiment
- Codex/cloud fallback

Replica mode is a baseline, not the product identity. Layer split and
profile-driven placement are the main research direction.
