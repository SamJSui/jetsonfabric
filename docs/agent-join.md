# Agent Join Flow

Adding a Jetson should eventually be a repeatable install flow:

```bash
curl -fsSL https://example.invalid/jetsonmesh/install.sh | bash
jetsonmesh-agent \
  --control-url http://beelink:52415 \
  --join-token "$JETSONMESH_JOIN_TOKEN" \
  --node-id jetson-02
```

The control plane should reject unknown nodes unless they present a valid join
token.

## Expected Node Lifecycle

1. Agent starts.
2. Agent detects platform and runtimes.
3. Agent posts heartbeat.
4. Control plane records node as `online`.
5. Benchmark service runs a small probe.
6. Placement planner marks compatible models as candidates.
7. Scheduler may assign work to the node.

## Why This Matters

This keeps expansion simple:

- buy a new Jetson
- flash JetPack
- install agent
- provide join token
- run calibration benchmarks
- node becomes eligible for routing

