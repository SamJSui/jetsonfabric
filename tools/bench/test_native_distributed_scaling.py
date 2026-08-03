import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.bench import native_distributed_scaling as scaling


class NativeDistributedScalingTest(unittest.TestCase):
    def test_manifest_rejects_layer_sum_mismatch(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "request.json").write_text("{}", encoding="utf-8")
            manifest = self.manifest()
            manifest["models"][0]["stage_layer_counts"] = [12, 15]
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sum to 27, want 28"):
                scaling.load_manifest(path)

    def test_manifest_resolves_request_relative_to_manifest(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            request = directory / "request.json"
            request.write_text("{}", encoding="utf-8")
            path = directory / "manifest.json"
            path.write_text(json.dumps(self.manifest()), encoding="utf-8")
            loaded = scaling.load_manifest(path)
            self.assertEqual(loaded["request_template"], request.resolve())

    def test_model_selection_rejects_unknown_labels(self):
        models = self.manifest()["models"]
        with self.assertRaisesRegex(ValueError, "unknown model labels: 7b"):
            scaling.select_models(models, {"1.5b", "7b"})

    def test_validate_active_deployment_requires_exact_ranges_and_hosts(self):
        model = self.manifest()["models"][0]
        active = scaling.validate_active_deployment(self.deployment(), model)
        self.assertEqual(active["deployment_id"], "deployment-a")
        invalid = self.deployment()
        invalid["active"]["stages"][1]["physical_host_id"] = "dopey"
        with self.assertRaisesRegex(RuntimeError, "distinct physical hosts"):
            scaling.validate_active_deployment(invalid, model)

    def test_collect_residency_requires_exact_weight_sum(self):
        client = FakeClient(
            {
                "http://dopey/v1/runtime/deployment": self.runtime_status(0, 18, 60),
                "http://grumpy/v1/runtime/deployment": self.runtime_status(18, 28, 40),
            }
        )
        result = scaling.collect_residency(client, self.deployment()["active"])
        self.assertEqual(result["resident_weight_bytes"], 100)
        client.responses["http://grumpy/v1/runtime/deployment"]["model_memory"][
            "resident_weight_bytes"
        ] = 39
        with self.assertRaisesRegex(RuntimeError, "sum to 99, want 100"):
            scaling.collect_residency(client, self.deployment()["active"])

    def test_correctness_requires_distributed_native_execution(self):
        result = {
            "plan": {"topology": "distributed", "physical_host_count": 2},
            "runtime_identity": {
                "deployment_id": "deployment-a",
                "engine": "native",
                "model_id": "qwen-native",
            },
            "result": {
                "sampled_tokens": [11],
                "generated_text": ",",
                "stages": [
                    {
                        "phase": "prefill",
                        "decode_step": 0,
                        "stage_index": 0,
                        "payload_kind_out": "activation",
                        "payload_out": 16,
                        "payload_crc32_out": 7,
                    },
                    {
                        "phase": "prefill",
                        "decode_step": 0,
                        "stage_index": 1,
                        "payload_kind_in": "activation",
                        "payload_in": 16,
                        "payload_crc32_in": 7,
                        "payload_kind_out": "sampled_token",
                    },
                ],
            },
        }
        validated = scaling.validate_correctness_result(
            result, self.deployment()["active"], self.manifest()["models"][0], [11]
        )
        self.assertEqual(validated["sampled_tokens"], [11])
        result["plan"]["topology"] = "colocated"
        with self.assertRaisesRegex(RuntimeError, "distributed hosts"):
            scaling.validate_correctness_result(
                result,
                self.deployment()["active"],
                self.manifest()["models"][0],
                [11],
            )

    def test_benchmark_command_uses_streaming_and_explicit_counts(self):
        command = scaling.benchmark_command(
            Path("jf-bench"),
            "http://dopey:52415/",
            Path("request.json"),
            Path("result.json"),
            2,
            10,
            2,
        )
        self.assertIn("--stream", command)
        self.assertEqual(command[command.index("--warmup") + 1], "2")
        self.assertEqual(command[command.index("--count") + 1], "10")
        self.assertEqual(command[command.index("--concurrency") + 1], "2")
        self.assertEqual(
            command[command.index("--url") + 1],
            "http://dopey:52415/v1/chat/completions",
        )

    def test_trace_chain_rejects_crc_mismatch(self):
        traces = [
            {
                "phase": "prefill",
                "decode_step": 0,
                "stage_index": 0,
                "payload_kind_out": "activation",
                "payload_out": 16,
                "payload_crc32_out": 7,
            },
            {
                "phase": "prefill",
                "decode_step": 0,
                "stage_index": 1,
                "payload_kind_in": "activation",
                "payload_in": 16,
                "payload_crc32_in": 8,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "CRCs do not match"):
            scaling.validate_trace_chain(traces, stage_count=2, generated_tokens=1)

    def test_benchmark_result_rejects_zero_success(self):
        result = {
            "success_count": 0,
            "failure_count": 2,
            "results": [{"error": "runtime exited"}, {"error": "runtime exited"}],
        }
        with self.assertRaisesRegex(RuntimeError, "C2 benchmark completed 0/2"):
            scaling.validate_benchmark_result(result, expected_count=2, concurrency=2)

    def test_benchmark_result_accepts_complete_streaming_run(self):
        result = {
            "success_count": 2,
            "failure_count": 0,
            "output_tokens": 256,
            "output_token_throughput": 12.5,
            "ttft": {"p50_ms": 100},
        }
        scaling.validate_benchmark_result(result, expected_count=2, concurrency=2)

    @staticmethod
    def manifest():
        return {
            "schema_version": 1,
            "name": "test",
            "request_template": "request.json",
            "warmups": 2,
            "requests": 10,
            "concurrencies": [1, 2],
            "correctness": {
                "prompt": "Once upon a time",
                "max_tokens": 1,
            },
            "models": [
                {
                    "label": "1.5b",
                    "model_id": "qwen-native",
                    "layer_count": 28,
                    "stage_layer_counts": [18, 10],
                    "context_size": 1536,
                    "expected_tokens": [11],
                }
            ],
        }

    @staticmethod
    def deployment():
        return {
            "phase": "active",
            "active": {
                "deployment_id": "deployment-a",
                "epoch": 2,
                "model": {"model_id": "qwen-native", "engine": "native"},
                "stages": [
                    {
                        "stage_index": 0,
                        "layer_start": 0,
                        "layer_end": 18,
                        "physical_host_id": "dopey",
                        "api_url": "http://dopey",
                    },
                    {
                        "stage_index": 1,
                        "layer_start": 18,
                        "layer_end": 28,
                        "physical_host_id": "grumpy",
                        "api_url": "http://grumpy",
                    },
                ],
            },
        }

    @staticmethod
    def runtime_status(start, end, resident):
        return {
            "active": True,
            "state": "active",
            "deployment": {"deployment_id": "deployment-a", "epoch": 2},
            "model_memory": {
                "layer_start": start,
                "layer_end": end,
                "resident_weight_bytes": resident,
                "total_weight_bytes": 100,
                "partitioned": True,
                "pinned": True,
            },
        }


class FakeClient:
    def __init__(self, responses):
        self.responses = responses

    def get_absolute(self, url):
        return self.responses[url]


if __name__ == "__main__":
    unittest.main()
