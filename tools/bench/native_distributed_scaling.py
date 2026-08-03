#!/usr/bin/env python3

"""Run a checkpointed native distributed-serving model scaling matrix."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA_VERSION = 1


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_keys(value, allowed, context):
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {', '.join(unknown)}")


def positive_int(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require_keys(
        manifest,
        {
            "schema_version",
            "name",
            "request_template",
            "warmups",
            "requests",
            "concurrencies",
            "correctness",
            "models",
        },
        "manifest",
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not str(manifest.get("name", "")).strip():
        raise ValueError("manifest name is required")
    request_template = (path.parent / manifest.get("request_template", "")).resolve()
    if not request_template.is_file():
        raise ValueError(f"request template does not exist: {request_template}")
    manifest["request_template"] = request_template
    positive_int(manifest.get("warmups"), "warmups")
    positive_int(manifest.get("requests"), "requests")
    concurrencies = manifest.get("concurrencies")
    if not isinstance(concurrencies, list) or not concurrencies:
        raise ValueError("concurrencies must be a nonempty list")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("concurrencies must be unique")
    for value in concurrencies:
        positive_int(value, "concurrency")
    validate_correctness(manifest.get("correctness"))
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    labels = set()
    for model in models:
        validate_model(model)
        if model["label"] in labels:
            raise ValueError(f"duplicate model label: {model['label']}")
        labels.add(model["label"])
    return manifest


def validate_correctness(correctness):
    if not isinstance(correctness, dict):
        raise ValueError("correctness must be an object")
    require_keys(correctness, {"prompt", "max_tokens"}, "correctness")
    if not str(correctness.get("prompt", "")).strip():
        raise ValueError("correctness prompt is required")
    positive_int(correctness.get("max_tokens"), "correctness max_tokens")


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError("each model must be an object")
    require_keys(
        model,
        {
            "label",
            "model_id",
            "layer_count",
            "stage_layer_counts",
            "context_size",
            "expected_tokens",
        },
        "model",
    )
    for field in ("label", "model_id"):
        if not str(model.get(field, "")).strip():
            raise ValueError(f"model {field} is required")
    layer_count = positive_int(model.get("layer_count"), "model layer_count")
    positive_int(model.get("context_size"), "model context_size")
    counts = model.get("stage_layer_counts")
    if not isinstance(counts, list) or len(counts) < 2:
        raise ValueError("stage_layer_counts must contain at least two stages")
    for count in counts:
        positive_int(count, "stage layer count")
    if sum(counts) != layer_count:
        raise ValueError(
            f"model {model['label']} stage layers sum to {sum(counts)}, want {layer_count}"
        )
    expected = model.get("expected_tokens")
    if expected is not None and (
        not isinstance(expected, list)
        or not expected
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in expected)
    ):
        raise ValueError("model expected_tokens must contain non-negative token IDs")


def select_models(models, selected_labels):
    if selected_labels is None:
        return models
    known_labels = {model["label"] for model in models}
    unknown_labels = sorted(selected_labels - known_labels)
    if unknown_labels:
        raise ValueError(f"unknown model labels: {', '.join(unknown_labels)}")
    return [model for model in models if model["label"] in selected_labels]


class JsonClient:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path):
        return self._request("GET", path, None)

    def post(self, path, body):
        return self._request("POST", path, body)

    def get_absolute(self, url):
        return self._request("GET", url, None, absolute=True)

    def _request(self, method, path, body, absolute=False):
        url = path if absolute else self.base_url + "/" + path.lstrip("/")
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} returned HTTP {error.code}: {detail}") from error


def stage_ranges(counts):
    ranges = []
    start = 0
    for count in counts:
        ranges.append((start, start + count))
        start += count
    return ranges


def validate_active_deployment(response, model):
    active = response.get("active") or {}
    stages = active.get("stages") or []
    expected_ranges = stage_ranges(model["stage_layer_counts"])
    if response.get("phase") not in ("active", "draining"):
        raise RuntimeError(f"deployment did not become active: {response.get('phase')}")
    if active.get("model", {}).get("model_id") != model["model_id"]:
        raise RuntimeError("coordinator activated the wrong model")
    if active.get("model", {}).get("engine") != "native":
        raise RuntimeError("coordinator did not select the native engine")
    if len(stages) != len(expected_ranges):
        raise RuntimeError("coordinator returned the wrong stage count")
    hosts = set()
    for index, (stage, expected) in enumerate(zip(stages, expected_ranges)):
        actual = (stage.get("layer_start"), stage.get("layer_end"))
        if stage.get("stage_index") != index or actual != expected:
            raise RuntimeError(f"stage {index} range is {actual}, want {expected}")
        hosts.add(stage.get("physical_host_id") or stage.get("hostname"))
    if None in hosts or len(hosts) != len(stages):
        raise RuntimeError("deployment stages are not on distinct physical hosts")
    return active


def collect_residency(client, active):
    rows = []
    for stage in active["stages"]:
        status = client.get_absolute(stage["api_url"].rstrip("/") + "/v1/runtime/deployment")
        deployment = status.get("deployment") or {}
        memory = status.get("model_memory") or {}
        if not status.get("active") or status.get("state") != "active":
            raise RuntimeError(f"stage {stage['stage_index']} runtime is not active")
        if deployment.get("deployment_id") != active["deployment_id"]:
            raise RuntimeError(f"stage {stage['stage_index']} has the wrong deployment")
        if deployment.get("epoch") != active["epoch"]:
            raise RuntimeError(f"stage {stage['stage_index']} has the wrong epoch")
        if (memory.get("layer_start"), memory.get("layer_end")) != (
            stage["layer_start"],
            stage["layer_end"],
        ):
            raise RuntimeError(f"stage {stage['stage_index']} residency range is wrong")
        if not memory.get("partitioned") or not memory.get("pinned"):
            raise RuntimeError(f"stage {stage['stage_index']} residency is not partitioned and pinned")
        rows.append({"stage": stage, "model_memory": memory})
    totals = {row["model_memory"].get("total_weight_bytes") for row in rows}
    if len(totals) != 1 or None in totals:
        raise RuntimeError("stage runtimes disagree on total model weight bytes")
    resident = sum(row["model_memory"].get("resident_weight_bytes", -1) for row in rows)
    total = totals.pop()
    if resident != total:
        raise RuntimeError(f"resident stage weights sum to {resident}, want {total}")
    return {"stages": rows, "resident_weight_bytes": resident, "total_weight_bytes": total}


def validate_correctness_result(result, active, model, expected_tokens):
    plan = result.get("plan") or {}
    output = result.get("result") or {}
    identity = result.get("runtime_identity") or {}
    if plan.get("topology") != "distributed" or plan.get("physical_host_count", 0) < 2:
        raise RuntimeError("correctness request did not execute on distributed hosts")
    if identity.get("deployment_id") != active["deployment_id"]:
        raise RuntimeError("correctness request used the wrong deployment")
    if identity.get("engine") != "native" or identity.get("model_id") != model["model_id"]:
        raise RuntimeError("correctness request used the wrong runtime identity")
    tokens = output.get("sampled_tokens") or []
    if not tokens:
        raise RuntimeError("correctness request did not sample a token")
    if expected_tokens is not None and tokens != expected_tokens:
        raise RuntimeError(f"sampled tokens {tokens} do not match expected {expected_tokens}")
    traces = output.get("stages") or []
    validate_trace_chain(traces, len(active["stages"]), len(tokens))
    return {
        "sampled_tokens": tokens,
        "generated_text": output.get("generated_text", ""),
        "prompt_tokens": output.get("prompt_tokens"),
        "completion_tokens": output.get("completion_tokens"),
        "traces": traces,
    }


def validate_trace_chain(traces, stage_count, generated_tokens):
    prefill = [trace for trace in traces if trace.get("phase") == "prefill"]
    decode = [trace for trace in traces if trace.get("phase") == "decode"]
    if len(prefill) != stage_count:
        raise RuntimeError("correctness trace has the wrong prefill stage count")
    if len(decode) != max(0, generated_tokens - 1) * stage_count:
        raise RuntimeError("correctness trace has the wrong decode stage count")
    activations = [trace for trace in traces if trace.get("payload_kind_out") == "activation"]
    if not activations:
        raise RuntimeError("correctness trace did not produce an inter-stage activation")
    for source in activations:
        match = next(
            (
                target
                for target in traces
                if target.get("phase") == source.get("phase")
                and target.get("decode_step") == source.get("decode_step")
                and target.get("stage_index") == source.get("stage_index") + 1
                and target.get("payload_kind_in") == "activation"
            ),
            None,
        )
        if match is None:
            raise RuntimeError("correctness trace activation has no receiving stage")
        if match.get("payload_in") != source.get("payload_out"):
            raise RuntimeError("correctness trace activation byte counts do not match")
        if match.get("payload_crc32_in") != source.get("payload_crc32_out"):
            raise RuntimeError("correctness trace activation CRCs do not match")


def benchmark_command(bench_bin, endpoint, request, output, warmups, count, concurrency):
    return [
        str(bench_bin),
        "--url",
        endpoint.rstrip("/") + "/v1/chat/completions",
        "--request",
        str(request),
        "--stream",
        "--warmup",
        str(warmups),
        "--count",
        str(count),
        "--concurrency",
        str(concurrency),
        "--output",
        str(output),
    ]


def run_benchmark(bench_bin, endpoint, template, model_id, warmups, count, concurrency):
    request = json.loads(template.read_text(encoding="utf-8"))
    request["model"] = model_id
    request["stream"] = True
    with tempfile.TemporaryDirectory(prefix="jf-native-scaling-") as directory:
        directory = Path(directory)
        request_path = directory / "request.json"
        output_path = directory / "result.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        command = benchmark_command(
            bench_bin, endpoint, request_path, output_path, warmups, count, concurrency
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"jf-bench failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def run_model(client, args, manifest, model):
    deployment = client.post(
        "/v1/deployments/switch",
        {
            "deployment_id": f"{args.run_id}-{model['label']}",
            "model": model["model_id"],
            "stage_count": len(model["stage_layer_counts"]),
            "stage_layer_counts": model["stage_layer_counts"],
            "ctx_size": model["context_size"],
        },
    )
    active = validate_active_deployment(deployment, model)
    residency = collect_residency(client, active)
    correctness_spec = manifest["correctness"]
    correctness = client.post(
        "/v1/layer-split/run",
        {
            "request_id": f"{args.run_id}-{model['label']}-correctness",
            "model": model["model_id"],
            "payload": correctness_spec["prompt"],
            "max_tokens": correctness_spec["max_tokens"],
            "stage_count": len(model["stage_layer_counts"]),
            "allow_colocated_stages": False,
        },
    )
    correctness = validate_correctness_result(
        correctness, active, model, model.get("expected_tokens")
    )
    benchmarks = {}
    for concurrency in manifest["concurrencies"]:
        benchmarks[f"c{concurrency}"] = run_benchmark(
            args.bench_bin,
            args.coordinator_url,
            manifest["request_template"],
            model["model_id"],
            manifest["warmups"],
            manifest["requests"],
            concurrency,
        )
    return {
        "label": model["label"],
        "model_id": model["model_id"],
        "status": "complete",
        "deployment": deployment,
        "residency": residency,
        "correctness": correctness,
        "benchmarks": benchmarks,
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--bench-bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="native-scaling")
    parser.add_argument("--models", help="comma-separated model labels")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.bench_bin = args.bench_bin.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.bench_bin.is_file():
        parser.error(f"benchmark binary does not exist: {args.bench_bin}")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.selected_models = (
        {part.strip() for part in args.models.split(",") if part.strip()}
        if args.models
        else None
    )
    return args


def main(argv=None):
    args = arguments(argv)
    manifest = load_manifest(args.manifest)
    models = select_models(manifest["models"], args.selected_models)
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": manifest["name"],
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "coordinator_url": args.coordinator_url,
        "manifest_sha256": sha256_file(args.manifest),
        "bench_binary_sha256": sha256_file(args.bench_bin),
        "workload": {
            "request_template_sha256": sha256_file(manifest["request_template"]),
            "warmups": manifest["warmups"],
            "requests": manifest["requests"],
            "concurrencies": manifest["concurrencies"],
        },
        "cluster": None,
        "models": [],
    }
    write_report(report, args.output)
    client = JsonClient(args.coordinator_url, args.timeout)
    try:
        report["cluster"] = client.get("/v1/cluster/members")
        for model in models:
            try:
                report["models"].append(run_model(client, args, manifest, model))
            except Exception as error:
                report["models"].append(
                    {
                        "label": model["label"],
                        "model_id": model["model_id"],
                        "status": "failed",
                        "error": str(error),
                    }
                )
                write_report(report, args.output)
                if not args.continue_on_error:
                    raise
            write_report(report, args.output)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["finished_at"] = utc_now()
        write_report(report, args.output)
        raise
    report["status"] = (
        "complete" if all(model["status"] == "complete" for model in report["models"])
        else "partial"
    )
    report["finished_at"] = utc_now()
    write_report(report, args.output)


if __name__ == "__main__":
    main()
