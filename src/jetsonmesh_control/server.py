from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass
class NodeRecord:
    node_id: str
    hostname: str
    arch: str
    os: str
    capabilities: dict[str, Any]
    metrics: dict[str, Any]
    last_seen: float = field(default_factory=time.time)


class ControlState:
    def __init__(self, join_token: str, model_registry_path: Path) -> None:
        self.join_token = join_token
        self.nodes: dict[str, NodeRecord] = {}
        self.models = self._load_models(model_registry_path)

    @staticmethod
    def _load_models(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("models", [])

    def upsert_node(self, payload: dict[str, Any]) -> NodeRecord:
        record = NodeRecord(
            node_id=payload["node_id"],
            hostname=payload.get("hostname", payload["node_id"]),
            arch=payload.get("arch", "unknown"),
            os=payload.get("os", "unknown"),
            capabilities=payload.get("capabilities", {}),
            metrics=payload.get("metrics", {}),
        )
        self.nodes[record.node_id] = record
        return record


class Handler(BaseHTTPRequestHandler):
    state: ControlState

    def _send_json(self, status: int, payload: dict[str, Any] | list[Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _authorize(self) -> bool:
        expected = self.state.join_token
        if not expected:
            return True
        auth_header = self.headers.get("Authorization", "")
        return auth_header == f"Bearer {expected}"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(200, {"status": "ok", "service": "jetsonmesh-control"})
            return
        if parsed.path == "/v1/nodes":
            nodes = [asdict(node) for node in self.state.nodes.values()]
            self._send_json(200, {"nodes": nodes})
            return
        if parsed.path == "/v1/models":
            self._send_json(200, {"models": self.state.models})
            return
        if parsed.path == "/v1/routes/preview":
            query = parse_qs(parsed.query)
            model_id = query.get("model", [""])[0]
            self._send_json(200, preview_route(model_id, self.state))
            return
        self._send_json(404, {"error": "not_found", "path": parsed.path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/agents/heartbeat":
            if not self._authorize():
                self._send_json(401, {"error": "unauthorized"})
                return
            payload = self._read_json()
            if "node_id" not in payload:
                self._send_json(400, {"error": "missing_node_id"})
                return
            record = self.state.upsert_node(payload)
            self._send_json(200, {"status": "registered", "node": asdict(record)})
            return
        if parsed.path == "/v1/chat/completions":
            self._send_json(
                501,
                {
                    "error": "not_implemented",
                    "message": "Model routing is scaffolded but no runtime backend is wired yet.",
                    "planned_router_inputs": [
                        "model",
                        "latency_budget_ms",
                        "quality_floor",
                        "node_queue_depth",
                        "node_temperature",
                        "model_fit",
                    ],
                },
            )
            return
        self._send_json(404, {"error": "not_found", "path": parsed.path})

    def log_message(self, format: str, *args: Any) -> None:
        # Keep local development output compact.
        return


def preview_route(model_id: str, state: ControlState) -> dict[str, Any]:
    models_by_id = {model["id"]: model for model in state.models}
    model = models_by_id.get(model_id)
    if model is None:
        return {"model": model_id, "valid": False, "reason": "unknown_model"}

    placements = []
    for node in state.nodes.values():
        node_caps = node.capabilities
        min_memory_gb = model.get("min_memory_gb", 0)
        node_memory_gb = node_caps.get("memory_gb", 0)
        required_accelerator = model.get("preferred_accelerator")
        has_accelerator = required_accelerator in node_caps.get("accelerators", [])
        memory_ok = node_memory_gb >= min_memory_gb
        accelerator_ok = not required_accelerator or has_accelerator
        placements.append(
            {
                "node_id": node.node_id,
                "valid": memory_ok and accelerator_ok,
                "memory_ok": memory_ok,
                "accelerator_ok": accelerator_ok,
                "reason": route_reason(memory_ok, accelerator_ok, required_accelerator),
            }
        )

    return {"model": model_id, "valid": True, "placements": placements}


def route_reason(memory_ok: bool, accelerator_ok: bool, accelerator: str | None) -> str:
    if not memory_ok:
        return "insufficient_memory"
    if not accelerator_ok:
        return f"missing_accelerator:{accelerator}"
    return "candidate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JetsonMesh control plane.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=52415)
    parser.add_argument("--join-token", default="dev-token")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "models.example.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Handler.state = ControlState(args.join_token, args.models)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"JetsonMesh control plane listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

