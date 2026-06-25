# Agent Join Flow

Adding a Jetson should eventually be a repeatable install flow:

```bash
curl -fsSL https://example.invalid/jetsonmesh/install.sh | bash
./jetsonmesh-agent \
  --control-url http://beelink:52415 \
  --join-token "$JETSONMESH_JOIN_TOKEN" \
  --node-id jetson-02 \
  --llama-url http://127.0.0.1:8080 \
  --llama-models qwen2.5-coder-1.5b-q4
```

The control plane should reject unknown nodes unless they present a valid join
token.

## Expected Node Lifecycle

1. Agent starts.
2. Agent detects platform and runtimes.
3. Agent advertises local runtime backend URLs when configured.
4. Agent posts heartbeat.
5. Control plane records node as `online`.
6. Benchmark service runs a small probe.
7. Placement planner marks compatible models as candidates.
8. Scheduler may assign work to the node.

## Why This Matters

This keeps expansion simple:

- buy a new Jetson
- flash JetPack
- install agent
- provide join token
- run calibration benchmarks
- node becomes eligible for routing
