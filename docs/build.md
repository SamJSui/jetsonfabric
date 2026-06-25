# Build Process

JetsonMesh runtime services are Go-native. Python is reserved for later
benchmark analysis, plotting, and report notebooks.

## Toolchain

- Go 1.26.4 or newer
- Linux arm64 target for Jetson Orin nodes
- Optional C++ toolchain later for optimized shard/runtime adapters

On the Windows development machine, the local Go toolchain is installed at:

```text
C:\Users\sui\Documents\tools\go\bin\go.exe
```

## Local Build

From the repo root:

```powershell
.\scripts\build.ps1
```

This runs tests and builds:

- `dist\jetsonmesh-control-windows-amd64.exe`
- `dist\jetsonmesh-agent-windows-amd64.exe`
- `dist\jetsonmesh-control-linux-arm64`
- `dist\jetsonmesh-agent-linux-arm64`

## Development Run

Control plane:

```powershell
.\scripts\run-control.ps1
```

Agent:

```powershell
.\scripts\run-agent.ps1 -NodeId dev-node
```

## Jetson Agent Install Target

The intended Jetson deployment path is:

1. Cross-compile `jetsonmesh-agent-linux-arm64`.
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

