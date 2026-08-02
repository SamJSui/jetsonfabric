#!/usr/bin/env python3
"""Benchmark JetsonFabric's runtime JSONL generation endpoint."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestResult:
    latency_ms: float
    ttft_ms: float
    inter_token_latency_ms: list[float]
    sampled_tokens: list[int]
    generated_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    events: list[dict[str, Any]]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def request_payload(template: dict[str, Any], run_id: str) -> bytes:
    payload = copy.deepcopy(template)
    payload["request_id"] = f"runtime-bench-request-{run_id}"
    payload["session_id"] = f"runtime-bench-session-{run_id}"
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def run_request(
    url: str,
    template: dict[str, Any],
    timeout: float,
    cluster_token: str = "",
) -> RequestResult:
    run_id = uuid.uuid4().hex
    headers = {"Content-Type": "application/json"}
    if cluster_token:
        headers["X-JetsonFabric-Cluster-Token"] = cluster_token
    request = urllib.request.Request(
        url,
        data=request_payload(template, run_id),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter_ns()
    token_times: list[int] = []
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    done: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"runtime returned HTTP {response.status}")
            for raw_line in response:
                line = raw_line.strip()
                if not line:
                    continue
                observed = time.perf_counter_ns()
                event = json.loads(line)
                events.append(event)
                if event.get("type") == "error":
                    raise RuntimeError(f"{event.get('code')}: {event.get('message')}")
                if event.get("type") == "token":
                    token_times.append(observed)
                    text_parts.append(str(event.get("text", "")))
                elif event.get("type") == "done":
                    done = event
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"runtime returned HTTP {error.code}: {detail}") from error

    finished = time.perf_counter_ns()
    if done is None:
        raise RuntimeError("runtime stream ended without a done event")
    sampled_tokens = [int(event["token"]) for event in events if event.get("type") == "token"]
    authoritative = [int(token) for token in done.get("sampled_tokens", [])]
    if authoritative != sampled_tokens:
        raise RuntimeError("token events do not match the done event")
    if int(done.get("completion_tokens", -1)) != len(sampled_tokens):
        raise RuntimeError("completion token count does not match token events")

    ttft_ms = (token_times[0] - started) / 1_000_000 if token_times else 0.0
    itl_ms = [
        (current - previous) / 1_000_000
        for previous, current in zip(token_times, token_times[1:])
    ]
    return RequestResult(
        latency_ms=(finished - started) / 1_000_000,
        ttft_ms=ttft_ms,
        inter_token_latency_ms=itl_ms,
        sampled_tokens=sampled_tokens,
        generated_text="".join(text_parts),
        finish_reason=str(done.get("finish_reason", "")),
        prompt_tokens=int(done.get("prompt_tokens", 0)),
        completion_tokens=int(done.get("completion_tokens", 0)),
        events=events,
    )


def summarize(results: list[RequestResult], elapsed_seconds: float) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    ttfts = [result.ttft_ms for result in results]
    itls = [value for result in results for value in result.inter_token_latency_ms]
    output_tokens = sum(result.completion_tokens for result in results)
    decode_intervals = sum(len(result.inter_token_latency_ms) for result in results)
    decode_seconds = sum(sum(result.inter_token_latency_ms) for result in results) / 1000.0
    return {
        "latency_ms": summarize_values(latencies),
        "ttft_ms": summarize_values(ttfts),
        "inter_token_latency_ms": summarize_values(itls),
        "output_tokens": output_tokens,
        "output_token_throughput": output_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "end_to_end_tokens_per_second": (
            output_tokens / (sum(latencies) / 1000.0) if latencies else 0.0
        ),
        "decode_tokens_per_second": (
            decode_intervals / decode_seconds if decode_seconds else 0.0
        ),
    }


def expected_tokens(path: pathlib.Path) -> list[int]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(decoded, list):
        return [int(token) for token in decoded]
    try:
        return [int(token) for token in decoded["results"][0]["sampled_tokens"]]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("expected token file must be an array or benchmark report") from error


def validate_equivalent_tokens(
    results: list[RequestResult],
    expected: list[int] | None = None,
) -> None:
    reference = expected if expected is not None else results[0].sampled_tokens
    for index, result in enumerate(results):
        if result.sampled_tokens != reference:
            raise RuntimeError(f"request {index} sampled tokens differ from the reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="runtime /v1/generate URL")
    parser.add_argument("--request", required=True, type=pathlib.Path, help="request JSON template")
    parser.add_argument("--count", type=int, default=5, help="measured sequential requests")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured requests")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")
    parser.add_argument("--name", default="runtime-jsonl", help="benchmark name")
    parser.add_argument("--output", type=pathlib.Path, help="write summary JSON to this path")
    parser.add_argument(
        "--expected-tokens",
        type=pathlib.Path,
        help="JSON token array or prior benchmark report required for equivalence",
    )
    args = parser.parse_args()
    if args.count <= 0 or args.warmup < 0 or args.timeout <= 0:
        parser.error("count and timeout must be positive; warmup must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    template = json.loads(args.request.read_text(encoding="utf-8"))
    cluster_token = os.environ.get("JETSONFABRIC_CLUSTER_TOKEN", "")
    for _ in range(args.warmup):
        run_request(args.url, template, args.timeout, cluster_token)

    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    results = [
        run_request(args.url, template, args.timeout, cluster_token)
        for _ in range(args.count)
    ]
    validate_equivalent_tokens(
        results,
        expected_tokens(args.expected_tokens) if args.expected_tokens else None,
    )
    elapsed = time.perf_counter() - started
    finished_at = dt.datetime.now(dt.timezone.utc)
    report = {
        "name": args.name,
        "url": args.url,
        "request_count": args.count,
        "warmup_count": args.warmup,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "summary": summarize(results, elapsed),
        "results": [asdict(result) for result in results],
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
