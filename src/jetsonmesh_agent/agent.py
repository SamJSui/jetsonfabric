from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


def detect_capabilities() -> dict[str, Any]:
    accelerators: list[str] = []
    tegrastats_available = command_exists("tegrastats")
    if tegrastats_available:
        accelerators.append("cuda")
        accelerators.append("jetson")

    return {
        "arch": platform.machine(),
        "memory_gb": read_memory_gb(),
        "accelerators": accelerators,
        "tegrastats": tegrastats_available,
        "runtimes": detect_runtimes(),
    }


def detect_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "load_average": read_load_average(),
        "temperature_c": None,
        "queue_depth": 0,
    }
    if command_exists("tegrastats"):
        metrics["jetson_hint"] = "tegrastats_available"
    return metrics


def detect_runtimes() -> list[str]:
    runtimes = []
    for command, name in [
        ("docker", "docker"),
        ("trtexec", "tensorrt"),
        ("llama-cli", "llama.cpp"),
        ("ollama", "ollama"),
    ]:
        if command_exists(command):
            runtimes.append(name)
    return runtimes


def command_exists(command: str) -> bool:
    try:
        subprocess.run(
            ["where" if os.name == "nt" else "which", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def read_memory_gb() -> float:
    if os.name == "nt":
        return 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
    except OSError:
        return 0.0
    return 0.0


def read_load_average() -> list[float] | None:
    if hasattr(os, "getloadavg"):
        return [round(value, 2) for value in os.getloadavg()]
    return None


def heartbeat_payload(node_id: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "os": platform.platform(),
        "capabilities": detect_capabilities(),
        "metrics": detect_metrics(),
    }


def send_heartbeat(control_url: str, join_token: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{control_url.rstrip('/')}/v1/agents/heartbeat",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {join_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a JetsonMesh node agent.")
    parser.add_argument("--control-url", default="http://127.0.0.1:52415")
    parser.add_argument("--join-token", default="dev-token")
    parser.add_argument("--node-id", default=socket.gethostname())
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    while True:
        payload = heartbeat_payload(args.node_id)
        try:
            send_heartbeat(args.control_url, args.join_token, payload)
            print(f"heartbeat sent: {args.node_id}")
        except urllib.error.URLError as exc:
            print(f"heartbeat failed: {exc}")
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

