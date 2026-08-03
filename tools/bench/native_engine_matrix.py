#!/usr/bin/env python3

"""Run matched JetsonFabric-native and llama.cpp greedy benchmarks."""

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def parse_int_list(value, name, allow_zero=False):
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"{name} values must be {qualifier}")
    return values


def expand_prompt(seed, length):
    if not seed:
        raise ValueError("prompt seed cannot be empty")
    repeats = math.ceil(length / len(seed))
    return (seed * repeats)[:length]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_json(command):
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        rendered = " ".join(str(part) for part in command)
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}): {rendered}\n{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark returned invalid JSON: {completed.stdout}") from error


def distribution(samples):
    if not samples:
        raise ValueError("benchmark sample list cannot be empty")
    mean = statistics.fmean(samples)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * standard_deviation / math.sqrt(len(samples))
    return {
        "count": len(samples),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "normal_95_ci": [mean - margin, mean + margin],
    }


def summarize(raw):
    return {
        "ttft_ms": distribution(raw["ttft_ms"]),
        "itl_ms": distribution(raw["itl_ms"]),
        "decode_tokens_per_second": distribution(raw["decode_tokens_per_second"]),
        "end_to_end_tokens_per_second": distribution(
            raw["end_to_end_tokens_per_second"]
        ),
    }


SAMPLE_FIELDS = (
    "ttft_ms",
    "itl_ms",
    "decode_tokens_per_second",
    "end_to_end_tokens_per_second",
    "alternate_ttft_ms",
    "alternate_end_to_end_tokens_per_second",
    "prefill_planning_ms",
    "prefill_allocation_ms",
    "prefill_host_input_preparation_ms",
    "prefill_compute_ms",
    "prefill_output_read_ms",
)


def pool_runs(runs):
    if not runs:
        raise ValueError("cannot pool an empty benchmark run list")
    pooled = dict(runs[0])
    for field in SAMPLE_FIELDS:
        if field in runs[0]:
            pooled[field] = [sample for run in runs for sample in run[field]]
    for field in (
        "prefill_plan_reuse_count",
        "prefill_attention_backend_verified_count",
        "iterations",
    ):
        if field in runs[0]:
            pooled[field] = sum(run[field] for run in runs)
    pooled["replicate_count"] = len(runs)
    pooled["warmups_per_replicate"] = runs[0]["warmups"]
    median_fields = {
        "ttft_p50_ms": "ttft_ms",
        "itl_p50_ms": "itl_ms",
        "decode_tokens_per_second_p50": "decode_tokens_per_second",
        "end_to_end_tokens_per_second_p50": "end_to_end_tokens_per_second",
        "prefill_planning_p50_ms": "prefill_planning_ms",
        "prefill_allocation_p50_ms": "prefill_allocation_ms",
        "prefill_host_input_preparation_p50_ms": "prefill_host_input_preparation_ms",
        "prefill_compute_p50_ms": "prefill_compute_ms",
        "prefill_output_read_p50_ms": "prefill_output_read_ms",
    }
    for median_field, sample_field in median_fields.items():
        if sample_field in pooled:
            pooled[median_field] = statistics.median(pooled[sample_field])
    return pooled


def percent_change(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def compare(native, baseline):
    return {
        "ttft_change_percent": percent_change(
            native["ttft_p50_ms"], baseline["ttft_p50_ms"]
        ),
        "itl_reduction_percent": 100.0
        * (1.0 - native["itl_p50_ms"] / baseline["itl_p50_ms"]),
        "decode_throughput_change_percent": percent_change(
            native["decode_tokens_per_second_p50"],
            baseline["decode_tokens_per_second_p50"],
        ),
        "end_to_end_throughput_change_percent": percent_change(
            native["end_to_end_tokens_per_second_p50"],
            baseline["end_to_end_tokens_per_second_p50"],
        ),
    }


def compare_native_attention(flash, unfused):
    return {
        **compare(flash, unfused),
        "prefill_compute_change_percent": percent_change(
            flash["prefill_compute_p50_ms"], unfused["prefill_compute_p50_ms"]
        ),
        "prefill_scratch_change_percent": percent_change(
            flash["prefill_scratch_bytes"], unfused["prefill_scratch_bytes"]
        ),
    }


def validate_pair(native, llama, model_sha256, expected_prefill, expected_decode):
    if native["source_sha256"] != model_sha256:
        raise RuntimeError("JFM source hash does not match the benchmark GGUF")
    if native["prompt_tokens"] != llama["prompt_tokens"]:
        raise RuntimeError("native and llama.cpp prompt token IDs differ")
    if native["sampled_tokens"] != llama["sampled_tokens"]:
        raise RuntimeError("native and llama.cpp greedy token IDs differ")
    if native.get("session_policy") != "exact_shape_reuse_enabled":
        raise RuntimeError("native benchmark did not enable exact-shape session reuse")
    if native.get("prefill_attention_kernel") != expected_prefill:
        raise RuntimeError("native benchmark did not resolve the requested prefill kernel")
    if native.get("decode_attention_kernel") != expected_decode:
        raise RuntimeError("native benchmark did not resolve the requested decode kernel")
    verified = native.get("prefill_attention_backend_verified_count")
    expected_verified = native["iterations"] if expected_prefill == "flash" else 0
    if verified != expected_verified:
        raise RuntimeError("native benchmark did not prove the selected attention backend")


def tokens_argument(tokens):
    return ",".join(str(token) for token in tokens)


def run_attention_parity(args, prompt):
    result = run_json(
        [
            str(args.native_parity_bin),
            "--package",
            str(args.package),
            "--backend",
            args.backend,
            "--tokens",
            tokens_argument(prompt),
            "--threads",
            str(args.threads),
        ]
    )
    if not result.get("argmax_match"):
        raise RuntimeError("native flash and unfused prefill selected different tokens")
    if result.get("normalized_rmse", math.inf) > args.max_logit_nrmse:
        raise RuntimeError("native flash prefill exceeded the normalized logit error limit")
    if result.get("cosine_similarity", -math.inf) < args.min_logit_cosine:
        raise RuntimeError("native flash prefill fell below the logit cosine limit")
    return result


def native_command(args, prompt, output_length, attention_kernel):
    return [
        str(args.native_bin),
        "--package",
        str(args.package),
        "--backend",
        args.backend,
        "--tokens",
        tokens_argument(prompt),
        "--max-tokens",
        str(output_length),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--threads",
        str(args.threads),
        "--decode-policy",
        "incremental",
        "--session-policy",
        "warm",
        "--prefill-attention-kernel",
        attention_kernel,
        "--decode-attention-kernel",
        "flash",
    ]


def llama_command(args, prompt, output_length):
    return [
        str(args.llama_bin),
        "--model",
        str(args.model),
        "--tokens",
        tokens_argument(prompt),
        "--max-tokens",
        str(output_length),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--threads",
        str(args.threads),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
    ]


def run_case(args, prompt_length, output_length, model_sha256):
    prompt = expand_prompt(args.prompt_seed, prompt_length)
    commands = {
        "native": native_command(args, prompt, output_length, "flash"),
        "native_unfused": native_command(args, prompt, output_length, "unfused"),
        "llama.cpp": llama_command(args, prompt, output_length),
    }
    orders = [
        ["llama.cpp", "native_unfused", "native"],
        ["native", "llama.cpp", "native_unfused"],
        ["native_unfused", "native", "llama.cpp"],
    ]
    runs = {engine: [] for engine in commands}
    for order in orders:
        print(
            f"running prompt={prompt_length} output={output_length} "
            f"order={','.join(order)}",
            file=sys.stderr,
            flush=True,
        )
        current = {}
        for engine in order:
            current[engine] = run_json(commands[engine])
            runs[engine].append(current[engine])
        validate_pair(
            current["native"], current["llama.cpp"], model_sha256, "flash", "flash"
        )
        validate_pair(
            current["native_unfused"], current["llama.cpp"],
            model_sha256, "unfused", "flash"
        )
    results = {engine: pool_runs(engine_runs) for engine, engine_runs in runs.items()}
    return {
        "prompt_tokens": prompt_length,
        "output_tokens": output_length,
        "execution_orders": orders,
        "token_match": True,
        "native": results["native"],
        "native_unfused": results["native_unfused"],
        "llama_cpp": results["llama.cpp"],
        "statistics": {
            "native": summarize(results["native"]),
            "native_unfused": summarize(results["native_unfused"]),
            "llama_cpp": summarize(results["llama.cpp"]),
        },
        "comparison": compare(results["native"], results["llama.cpp"]),
        "flash_vs_unfused": compare_native_attention(
            results["native"], results["native_unfused"]
        ),
    }


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def existing_path(value):
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def write_report(report, output):
    rendered = json.dumps(report, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bin", required=True, type=existing_file)
    parser.add_argument("--native-parity-bin", required=True, type=existing_file)
    parser.add_argument("--llama-bin", required=True, type=existing_file)
    parser.add_argument("--package", required=True, type=existing_path)
    parser.add_argument("--model", required=True, type=existing_file)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--max-logit-nrmse", type=float, default=0.025)
    parser.add_argument("--min-logit-cosine", type=float, default=0.999)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--prompt-seed", default="12522,5193,264,882")
    parser.add_argument("--prompt-lengths", default="32,128,512,2048")
    parser.add_argument("--output-lengths", default="32,128,512")
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(argv)
    parsed.prompt_seed = parse_int_list(parsed.prompt_seed, "prompt seed", allow_zero=True)
    parsed.prompt_lengths = parse_int_list(parsed.prompt_lengths, "prompt lengths")
    parsed.output_lengths = parse_int_list(parsed.output_lengths, "output lengths")
    if parsed.threads <= 0 or parsed.iterations <= 1 or parsed.warmups <= 0:
        parser.error(
            "threads must be positive, iterations must exceed one, "
            "and matched comparisons require at least one warmup"
        )
    if parsed.n_gpu_layers < 0:
        parser.error("n-gpu-layers cannot be negative")
    if parsed.max_logit_nrmse < 0.0 or not -1.0 <= parsed.min_logit_cosine <= 1.0:
        parser.error("logit parity thresholds are outside their valid ranges")
    return parsed


def main(argv=None):
    args = arguments(argv)
    model_sha256 = sha256_file(args.model)
    started_at = dt.datetime.now(dt.timezone.utc)
    report = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "status": "running",
        "revision": args.revision,
        "model_sha256": model_sha256,
        "package_source_sha256": None,
        "executables": {
            "native": str(args.native_bin),
            "native_sha256": sha256_file(args.native_bin),
            "native_parity": str(args.native_parity_bin),
            "native_parity_sha256": sha256_file(args.native_parity_bin),
            "llama_cpp": str(args.llama_bin),
            "llama_cpp_sha256": sha256_file(args.llama_bin),
        },
        "workload": {
            "prompt_seed": args.prompt_seed,
            "prompt_lengths": args.prompt_lengths,
            "output_lengths": args.output_lengths,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "threads": args.threads,
            "backend": args.backend,
            "n_gpu_layers": args.n_gpu_layers,
            "sampling": "greedy",
            "comparison_scope": "matched_exact_shape_reuse",
            "native_session_policy": "exact_shape_reuse_enabled",
            "native_prefill_attention_kernel": "flash",
            "native_control_prefill_attention_kernel": "unfused",
            "native_decode_attention_kernel": "flash",
            "llama_context_policy": "reused_context_kv_cleared",
            "max_logit_nrmse": args.max_logit_nrmse,
            "min_logit_cosine": args.min_logit_cosine,
        },
        "attention_parity": [],
        "rows": [],
    }
    if args.output:
        write_report(report, args.output)
    try:
        for prompt_length in args.prompt_lengths:
            prompt = expand_prompt(args.prompt_seed, prompt_length)
            report["attention_parity"].append(run_attention_parity(args, prompt))
            for output_length in args.output_lengths:
                row = run_case(
                    args, prompt_length, output_length, model_sha256
                )
                report["rows"].append(row)
                report["package_source_sha256"] = row["native"]["source_sha256"]
                if args.output:
                    write_report(report, args.output)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if args.output:
            write_report(report, args.output)
        raise
    report["status"] = "complete"
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(report, args.output)


if __name__ == "__main__":
    main()
