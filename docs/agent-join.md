# Agent Join Flow

Adding a Jetson should eventually be a repeatable install flow:

```bash
curl -fsSL https://example.invalid/jetsonfabric/install.sh | bash
./jetsonfabric-agent \
  --control-url http://beelink:52415 \
  --join-token "$JETSONFABRIC_JOIN_TOKEN" \
  --node-name jetson-02 \
  --listen 0.0.0.0:52416 \
  --advertise-url http://jetson-02:52416 \
  --model-artifacts configs/model-artifacts.example.json \
  --llama-url http://127.0.0.1:8080 \
  --model qwen2.5-coder-1.5b-q4
```

The control plane should reject unknown nodes unless they present a valid join
token.

## Expected Node Lifecycle

1. Agent starts.
2. Agent loads model artifact metadata for the model it will advertise.
3. Agent detects platform and runtimes.
4. Agent starts its local proxy API.
5. Agent advertises the proxy URL and served model ID when configured.
6. Agent posts heartbeat.
7. Control plane records node as `online`.
8. Benchmark service runs a small probe.
9. Placement planner marks compatible models as candidates.
10. Scheduler may assign work to the node.

## Why This Matters

This keeps expansion simple:

- buy a new Jetson
- flash JetPack
- install agent
- provide join token
- run calibration benchmarks
- node becomes eligible for routing
