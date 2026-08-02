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


def percent_change(candidate, baseline):
    return 100.0 * (candidate / baseline - 1.0)


def compare(native, llama):
    return {
        "ttft_change_percent": percent_change(
            native["ttft_p50_ms"], llama["ttft_p50_ms"]
        ),
        "itl_reduction_percent": 100.0
        * (1.0 - native["itl_p50_ms"] / llama["itl_p50_ms"]),
        "decode_throughput_change_percent": percent_change(
            native["decode_tokens_per_second_p50"],
            llama["decode_tokens_per_second_p50"],
        ),
        "end_to_end_throughput_change_percent": percent_change(
            native["end_to_end_tokens_per_second_p50"],
            llama["end_to_end_tokens_per_second_p50"],
        ),
    }


def validate_pair(native, llama, model_sha256):
    if native["source_sha256"] != model_sha256:
        raise RuntimeError("JFM source hash does not match the benchmark GGUF")
    if native["prompt_tokens"] != llama["prompt_tokens"]:
        raise RuntimeError("native and llama.cpp prompt token IDs differ")
    if native["sampled_tokens"] != llama["sampled_tokens"]:
        raise RuntimeError("native and llama.cpp greedy token IDs differ")


def tokens_argument(tokens):
    return ",".join(str(token) for token in tokens)


def native_command(args, prompt, output_length):
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


def run_case(args, prompt_length, output_length, case_index, model_sha256):
    prompt = expand_prompt(args.prompt_seed, prompt_length)
    commands = {
        "native": native_command(args, prompt, output_length),
        "llama.cpp": llama_command(args, prompt, output_length),
    }
    order = ["llama.cpp", "native"] if case_index % 2 == 0 else ["native", "llama.cpp"]
    results = {}
    print(
        f"running prompt={prompt_length} output={output_length} order={','.join(order)}",
        file=sys.stderr,
        flush=True,
    )
    for engine in order:
        results[engine] = run_json(commands[engine])
    validate_pair(results["native"], results["llama.cpp"], model_sha256)
    return {
        "prompt_tokens": prompt_length,
        "output_tokens": output_length,
        "execution_order": order,
        "token_match": True,
        "native": results["native"],
        "llama_cpp": results["llama.cpp"],
        "statistics": {
            "native": summarize(results["native"]),
            "llama_cpp": summarize(results["llama.cpp"]),
        },
        "comparison": compare(results["native"], results["llama.cpp"]),
    }


def existing_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
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
    parser.add_argument("--llama-bin", required=True, type=existing_file)
    parser.add_argument("--package", required=True, type=existing_file)
    parser.add_argument("--model", required=True, type=existing_file)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
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
    if parsed.threads <= 0 or parsed.iterations <= 1 or parsed.warmups < 0:
        parser.error(
            "threads must be positive, iterations must exceed one, "
            "and warmups cannot be negative"
        )
    if parsed.n_gpu_layers < 0:
        parser.error("n-gpu-layers cannot be negative")
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
        },
        "rows": [],
    }
    if args.output:
        write_report(report, args.output)
    case_index = 0
    try:
        for prompt_length in args.prompt_lengths:
            for output_length in args.output_lengths:
                row = run_case(
                    args, prompt_length, output_length, case_index, model_sha256
                )
                report["rows"].append(row)
                report["package_source_sha256"] = row["native"]["source_sha256"]
                case_index += 1
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
