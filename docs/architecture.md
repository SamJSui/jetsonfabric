# Architecture

JetsonMesh has four layers:

1. Control plane
2. Node agents
3. Runtime workers
4. Benchmark and routing policy

## Control Plane

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

- llama.cpp for quantized LLM baselines
- TensorRT/ONNX for optimized inference
- Triton for production-style serving experiments
- custom layer-split runtime for distributed transformer inference

## Placement Modes

V0 supports route previews only. Planned modes:

- single-node
- replica baseline
- layer split
- tensor-parallel experiment
- Codex/cloud fallback

Replica mode is a baseline, not the product identity. Layer split and
profile-driven placement are the main research direction.

