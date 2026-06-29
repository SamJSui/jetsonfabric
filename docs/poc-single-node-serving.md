# POC - Single-Node Model Serving

The proof of concept is one full-model replica running on one Jetson through
JetsonFabric. This is not distributed inference yet. It is the baseline that
proves a node can be provisioned, can load a model, can serve a prompt, and can
be observed by the control plane.

Layer split, tensor parallelism, and multi-node scheduling need this baseline so
their later measurements have something honest to compare against.

## Goal

Run a small local model on one Jetson, route a request through
`jetsonfabric-control`, return the response through the JetsonFabric API, and record
a benchmark result.

## Why This Comes First

Without a real single-Jetson backend, distributed planning is mostly theoretical.
The POC creates the baseline needed to answer useful questions later:

- How fast is one Jetson for the selected model?
- What memory and thermal limits appear during real inference?
- Which runtime is easiest to operate from a Go agent/control plane?
- What does the route planner need to know before assigning work?
- What metric would a second Jetson need to improve?

## POC Deliverable

A demo should show:

1. `jetsonfabric-control` running on the Beelink or development machine.
2. `jetsonfabric-agent` running on one Jetson.
3. The control plane showing the Jetson as online.
4. A model backend running locally on that Jetson.
5. `/v1/chat/completions` routing one prompt to the Jetson agent, which proxies
   to that backend.
6. A response streamed or returned through JetsonFabric.
7. A benchmark record with latency, throughput, memory, and thermal data.

## Recommended First Backend

Use one existing local runtime first. Do not build a custom model engine for the
POC.

Preferred order:

1. llama.cpp server with a small quantized model.
2. Jetson AI Lab container if it is faster to get a working Jetson-optimized
   runtime.
3. Ollama only as a temporary convenience backend if it gets the first demo
   moving.

The Go control plane should treat the backend as an implementation detail behind
a small adapter interface.

## Candidate Models

Start with models small enough to fit comfortably on one Jetson:

- Qwen2.5-Coder 1.5B quantized
- Qwen2.5 1.5B quantized
- TinyLlama-class quantized baseline
- a small vision model later for TensorRT-oriented demos

The exact model can change, but the POC should avoid models that require fragile
swap-heavy execution. The point is to establish the serving path and metrics.

## Implementation Work

### Agent

- Detect Jetson hardware model.
- Detect JetPack/CUDA/TensorRT hints where available.
- Parse temperature and throttling signals.
- Report available memory.
- Report runtime availability such as `llama-server`, `trtexec`, Docker, or
  other local backends.
- Expose a small local proxy API for model requests.
- Keep node-local runtime URLs private to the agent when possible.

### Control Plane

- Store node runtime capabilities.
- Load model backend config from `configs/`.
- Select the single online compatible Jetson for the requested model.
- Implement the single-node path for `/v1/chat/completions`.
- Include route metadata in responses or logs.

### Runtime Adapter

- Add a Go client for the selected backend's local HTTP API.
- Normalize requests and responses into JetsonFabric API types.
- Track backend errors and timeouts.
- Keep the model process management simple at first; manual startup is
  acceptable for the first demo.
- The agent owns the local runtime URL, model ID served by that runtime, and
  the node-facing proxy endpoint. Model artifact download and runtime launch can
  remain script-managed until the first Jetson demo works.

### Benchmarks

- Record prompt, model ID, node name, route mode, start/end time, latency,
  time-to-first-token if available, output token count, tokens/sec, memory, and
  temperature.
- Store results as JSONL or another simple local file format first.
- Python may read benchmark output later for graphs and reports.

## Acceptance Criteria

The POC is done when these commands, or their documented equivalents, work on a
fresh setup:

```sh
sh scripts/build.sh
sh scripts/run-control.sh
```

And on the Jetson:

```bash
./jetsonfabric-agent \
  --control-url http://beelink:52415 \
  --join-token "$JETSONFABRIC_JOIN_TOKEN" \
  --node-name jetson-01 \
  --listen 0.0.0.0:52416 \
  --advertise-url http://jetson-01:52416 \
  --llama-url http://127.0.0.1:8080 \
  --model qwen2.5-coder-1.5b-q4
```

Then from the development machine:

```sh
curl -sS http://beelink:52415/v1/nodes

curl -sS \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder-1.5b-q4",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from the Jetson."
      }
    ]
  }' \
  http://beelink:52415/v1/chat/completions
```

The final response should come from the Jetson-hosted model backend, not from a
placeholder handler.

## Current Code Path

The POC serving path is:

```text
POST /v1/chat/completions
  -> control plane validates model and messages
  -> route selector finds one compatible online node
  -> node backend advertisement provides the agent proxy URL
  -> runtime client calls <agent-url>/v1/chat/completions
  -> agent proxies the request to its local <llama-url>/v1/chat/completions
  -> control plane attaches route metadata
  -> benchmark recorder writes data/benchmarks.jsonl
```

The backend can be a llama.cpp server or any temporary local server that speaks
the OpenAI chat-completions shape. The control plane should call the agent proxy,
not the runtime backend port directly.

## Explicitly Out Of Scope For The POC

- Layer-split inference.
- Tensor parallelism.
- Custom transformer runtime.
- Kubernetes.
- Multi-node scheduler optimization.
- Claims that JetsonFabric is faster than cloud or frontier models.
