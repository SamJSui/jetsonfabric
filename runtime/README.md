# JetsonFabric Runtime

This directory contains the future C++ runtime lane for JetsonFabric.

The first milestone is intentionally small: a runtime worker stub that exposes an OpenAI-compatible `/v1/chat/completions` endpoint and a `/healthz` endpoint. It exists to prove that the control/agent stack can route to a JetsonFabric-owned runtime instead of being coupled to `llama-server`.

## Build

```sh
make runtime-build
```

or:

```sh
sh scripts/build-runtime.sh
```

## Run Stub

```sh
make runtime-run
```

or:

```sh
LISTEN=127.0.0.1:9090 \
MODEL=qwen2.5-coder-1.5b-q4 \
MODE=single_node \
sh scripts/run-runtime-stub.sh
```

## Shape

- `worker/`: process entrypoint and CLI config.
- `api/`: minimal HTTP serving and response helpers.
- `engine/`: runtime execution abstraction and the current stub engine.

Future directories will hold tensor protocol, transport, CUDA helpers, and real model execution.
