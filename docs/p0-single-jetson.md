# P0 - Single Jetson Model Serving

P0 is the current priority. The goal is to get one real model working on one
Jetson through JetsonMesh before building distributed inference.

Layer split, tensor parallelism, and multi-node scheduling are later work. They
need a single-node baseline to compare against.

## Goal

Run a small local model on one Jetson, route a request through
`jetsonmesh-control`, return the response through the JetsonMesh API, and record
a benchmark result.

## Why This Comes First

Without a real single-Jetson backend, distributed planning is mostly theoretical.
P0 creates the baseline needed to answer useful questions later:

- How fast is one Jetson for the selected model?
- What memory and thermal limits appear during real inference?
- Which runtime is easiest to operate from a Go agent/control plane?
- What does the route planner need to know before assigning work?
- What metric would a second Jetson need to improve?

## P0 Deliverable

A demo should show:

1. `jetsonmesh-control` running on the Beelink or development machine.
2. `jetsonmesh-agent` running on one Jetson.
3. The control plane showing the Jetson as online.
4. A model backend running locally on that Jetson.
5. `/v1/chat/completions` routing one prompt to that backend.
6. A response streamed or returned through JetsonMesh.
7. A benchmark record with latency, throughput, memory, and thermal data.

## Recommended First Backend

Use one existing local runtime first. Do not build a custom model engine for P0.

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

The exact model can change, but P0 should avoid models that require fragile
swap-heavy execution. The point is to establish the serving path and metrics.

## Implementation Work

### Agent

- Detect Jetson hardware model.
- Detect JetPack/CUDA/TensorRT hints where available.
- Parse temperature and throttling signals.
- Report available memory.
- Report runtime availability such as `llama-server`, `trtexec`, Docker, or
  other local backends.

### Control Plane

- Store node runtime capabilities.
- Load model backend config from `configs/`.
- Select the single online compatible Jetson for the requested model.
- Implement the single-node path for `/v1/chat/completions`.
- Include route metadata in responses or logs.

### Runtime Adapter

- Add a Go client for the selected backend's local HTTP API.
- Normalize requests and responses into JetsonMesh API types.
- Track backend errors and timeouts.
- Keep the model process management simple at first; manual startup is
  acceptable for the first demo.

### Benchmarks

- Record prompt, model ID, node ID, route mode, start/end time, latency,
  time-to-first-token if available, output token count, tokens/sec, memory, and
  temperature.
- Store results as JSONL or another simple local file format first.
- Python may read benchmark output later for graphs and reports.

## Acceptance Criteria

P0 is done when these commands, or their documented equivalents, work on a fresh
setup:

```powershell
.\scripts\build.ps1
.\scripts\run-control.ps1
```

And on the Jetson:

```bash
./jetsonmesh-agent \
  --control-url http://beelink:52415 \
  --join-token "$JETSONMESH_JOIN_TOKEN" \
  --node-id jetson-01 \
  --llama-url http://127.0.0.1:8080 \
  --llama-models qwen2.5-coder-1.5b-q4
```

Then from the development machine:

```powershell
Invoke-WebRequest -UseBasicParsing http://beelink:52415/v1/nodes
$body = @{
  model = "qwen2.5-coder-1.5b-q4"
  messages = @(@{ role = "user"; content = "Say hello from the Jetson." })
} | ConvertTo-Json -Depth 5
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -Uri http://beelink:52415/v1/chat/completions
```

The final response should come from the Jetson-hosted model backend, not from a
placeholder handler.

## Current Code Path

The P0 serving path is now:

```text
POST /v1/chat/completions
  -> control plane validates model and messages
  -> route selector finds one compatible online node
  -> node backend advertisement provides an OpenAI-compatible llama URL
  -> runtime client calls <llama-url>/v1/chat/completions
  -> control plane attaches route metadata
  -> benchmark recorder writes data/benchmarks.jsonl
```

The backend can be a llama.cpp server or any temporary local server that speaks
the OpenAI chat-completions shape.

## Explicitly Out Of Scope For P0

- Layer-split inference.
- Tensor parallelism.
- Custom transformer runtime.
- Kubernetes.
- Multi-node scheduler optimization.
- Claims that JetsonMesh is faster than cloud or frontier models.
