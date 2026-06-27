# Build Process

JetsonFabric runtime services are Go-native. Python is reserved for later
benchmark analysis, plotting, and report notebooks.

## Toolchain

- Go 1.26.4 or newer
- WSL2 Ubuntu or another Linux development shell
- Linux arm64 target for Jetson Orin nodes
- Optional C++ toolchain later for optimized shard/runtime adapters

The scripts default to `go` from `PATH`. Set `GO=/path/to/go` if a specific Go
toolchain should be used.

When developing from WSL, install Go inside the WSL distribution and verify it
there:

```sh
go version
```

Do not depend on the Windows `go.exe` path from WSL for normal development.

## Local Build

From the repo root:

```sh
sh scripts/build.sh
```

This runs tests and builds:

- `dist/jetsonfabric-control-linux-amd64`
- `dist/jetsonfabric-agent-linux-amd64`
- `dist/jetsonfabric-control-linux-arm64`
- `dist/jetsonfabric-agent-linux-arm64`

## Development Run

Control plane:

```sh
sh scripts/run-control.sh
```

Agent:

```sh
sh scripts/run-agent.sh --node-id dev-node
```

## Jetson Agent Install Target

The intended Jetson deployment path is:

1. Cross-compile `jetsonfabric-agent-linux-arm64`.
2. Copy binary to Jetson.
3. Install a systemd unit.
4. Provide control-plane URL and join token.
5. Run calibration benchmarks before placing real model work.

## C++ Lane

C++ should be introduced for runtime-sensitive components, not for the control
plane:

- tensor/activation transport
- layer-shard runtime adapters
- TensorRT/ONNX execution wrappers
- pinned-buffer transfer experiments

