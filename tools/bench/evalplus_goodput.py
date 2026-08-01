#!/usr/bin/env python3
"""Compute correct EvalPlus completions per minute from normalized run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = {
    "humaneval": ("humaneval_passed", "humaneval_pass_at_1_percent"),
    "humaneval_plus": (
        "humaneval_plus_passed",
        "humaneval_plus_pass_at_1_percent",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, default="humaneval_plus")
    parser.add_argument("--min-pass-at-1", type=float, default=0.0)
    return parser.parse_args()


def run_duration_seconds(row: dict[str, Any]) -> float:
    for field in ("generation_duration_seconds", "duration_seconds"):
        value = row.get(field)
        if value is not None:
            return float(value)
    telemetry = row.get("humaneval_telemetry", {})
    durations = [
        float(node["duration_seconds"])
        for node in telemetry.values()
        if node.get("duration_seconds") is not None
    ]
    if not durations:
        raise ValueError(f"{row.get('label', row.get('model'))}: duration is missing")
    return max(durations)


def calculate(
    source: dict[str, Any],
    dataset: str,
    min_pass_at_1: float,
) -> dict[str, Any]:
    passed_field, percent_field = DATASETS[dataset]
    rows = []
    for row in source.get("rows", []):
        passed = int(row[passed_field])
        samples = int(row["samples"])
        percent = float(row[percent_field])
        duration = run_duration_seconds(row)
        if samples <= 0 or passed < 0 or passed > samples or duration <= 0:
            raise ValueError(f"{row.get('label', row.get('model'))}: invalid evaluation row")
        if percent < min_pass_at_1:
            continue
        rows.append(
            {
                "label": row.get("label"),
                "model": row.get("model"),
                "placement": row.get("placement"),
                "dataset": dataset,
                "samples": samples,
                "passed": passed,
                "pass_at_1_percent": percent,
                "duration_seconds": duration,
                "correct_completions_per_minute": passed * 60.0 / duration,
            }
        )
    return {
        "dataset": dataset,
        "min_pass_at_1_percent": min_pass_at_1,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = calculate(source, args.dataset, args.min_pass_at_1)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
