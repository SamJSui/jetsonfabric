# P2 - Operational Edge Fabric

P2 turns the measured runtime paths into a cohesive edge AI fabric. It does not
require tensor parallelism to succeed; it requires enough POC, layer-split, and
tensor-parallel evidence to know which paths are worth operating.

## Goal

Make JetsonFabric usable as an observable, repeatable, profile-driven edge
serving system.

## Scope

- Agent-managed model artifact download, verification, and runtime launch.
- Persistent control-plane state for nodes, deployments, benchmarks, and model
  metadata.
- Authenticated admin APIs for model and deployment management.
- Dashboard/API views for node health, route decisions, runtime readiness, and
  benchmark history.
- Profile-driven placement using memory, thermal, latency, and network data.
- Failover and degraded routing when a node or runtime disappears.
- Runtime transport optimization only where benchmarks identify a bottleneck:
  activation compression, pipelining, 10GbE TCP, pinned buffers, and eventually
  RDMA if justified.

## Why P2 Comes After The Runtime Evidence

POC, P0, and P1 answer what is technically possible and where the bottlenecks
are. P2 turns the useful measurements into product behavior: deploy, observe,
route, recover, and explain decisions without hand-tuning every run.
