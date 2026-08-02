import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.bench import native_engine_matrix


class NativeEngineMatrixTest(unittest.TestCase):
    def test_expand_prompt_repeats_and_truncates_seed(self):
        self.assertEqual(
            native_engine_matrix.expand_prompt([1, 2, 3], 8),
            [1, 2, 3, 1, 2, 3, 1, 2],
        )

    def test_parse_prompt_tokens_allows_zero(self):
        self.assertEqual(
            native_engine_matrix.parse_int_list("0,2", "prompt", allow_zero=True),
            [0, 2],
        )

    def test_existing_path_accepts_jfm_directory(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(
                native_engine_matrix.existing_path(directory),
                Path(directory).resolve(),
            )

    def test_distribution_preserves_count_and_mean(self):
        summary = native_engine_matrix.distribution([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertLess(summary["normal_95_ci"][0], 2.0)
        self.assertGreater(summary["normal_95_ci"][1], 2.0)

    def test_compare_reports_faster_native_decode(self):
        native = self.metrics(ttft=11.0, itl=8.0, decode=125.0, end_to_end=100.0)
        llama = self.metrics(ttft=10.0, itl=10.0, decode=100.0, end_to_end=80.0)
        comparison = native_engine_matrix.compare(native, llama)
        self.assertAlmostEqual(comparison["ttft_change_percent"], 10.0)
        self.assertAlmostEqual(comparison["itl_reduction_percent"], 20.0)
        self.assertAlmostEqual(comparison["decode_throughput_change_percent"], 25.0)
        self.assertAlmostEqual(
            comparison["end_to_end_throughput_change_percent"], 25.0
        )

    def test_pool_runs_recomputes_medians_and_counts(self):
        first = self.raw_run([1.0, 4.0], verified=2)
        second = self.raw_run([2.0, 3.0], verified=2)
        pooled = native_engine_matrix.pool_runs([first, second])
        self.assertEqual(pooled["ttft_ms"], [1.0, 4.0, 2.0, 3.0])
        self.assertEqual(pooled["ttft_p50_ms"], 2.5)
        self.assertEqual(pooled["iterations"], 4)
        self.assertEqual(pooled["prefill_attention_backend_verified_count"], 4)
        self.assertEqual(pooled["replicate_count"], 2)

    def test_validate_pair_rejects_token_mismatch(self):
        native = {
            "source_sha256": "abc",
            "prompt_tokens": [1],
            "sampled_tokens": [2],
            "session_policy": "exact_shape_reuse_enabled",
            "requested_backend": "cuda",
            "requested_prefill_attention_kernel": "flash",
            "prefill_attention_kernel": "flash",
            "decode_attention_kernel": "unfused",
            "prefill_attention_backend_verified_count": 2,
            "iterations": 2,
        }
        llama = {"prompt_tokens": [1], "sampled_tokens": [3]}
        with self.assertRaisesRegex(RuntimeError, "greedy token IDs differ"):
            native_engine_matrix.validate_pair(native, llama, "abc", "flash")

    def test_validate_pair_accepts_verified_flash_run(self):
        native = {
            "source_sha256": "abc",
            "prompt_tokens": [1],
            "sampled_tokens": [2],
            "session_policy": "exact_shape_reuse_enabled",
            "prefill_attention_kernel": "flash",
            "decode_attention_kernel": "unfused",
            "prefill_attention_backend_verified_count": 2,
            "iterations": 2,
        }
        llama = {"prompt_tokens": [1], "sampled_tokens": [2]}
        native_engine_matrix.validate_pair(native, llama, "abc", "flash")

    def test_native_command_enables_exact_shape_reuse(self):
        class Args:
            native_bin = Path("native")
            package = Path("model.jfm")
            backend = "cuda"
            warmups = 1
            iterations = 2
            threads = 6

        command = native_engine_matrix.native_command(Args(), [1, 2], 3, "flash")
        self.assertEqual(command[command.index("--session-policy") + 1], "warm")
        self.assertEqual(
            command[command.index("--prefill-attention-kernel") + 1], "flash"
        )
        self.assertEqual(
            command[command.index("--decode-attention-kernel") + 1], "unfused"
        )

    def test_run_attention_parity_rejects_excess_logit_error(self):
        class Args:
            native_parity_bin = Path("parity")
            package = Path("model.jfm")
            backend = "cuda"
            threads = 6
            max_logit_nrmse = 0.02
            min_logit_cosine = 0.999

        result = {
            "argmax_match": True,
            "normalized_rmse": 0.03,
            "cosine_similarity": 0.9999,
        }
        original = native_engine_matrix.run_json
        self.addCleanup(setattr, native_engine_matrix, "run_json", original)
        native_engine_matrix.run_json = lambda _command: result
        with self.assertRaisesRegex(RuntimeError, "normalized logit error"):
            native_engine_matrix.run_attention_parity(Args(), [1, 2])

    def test_run_attention_parity_rejects_low_cosine_similarity(self):
        class Args:
            native_parity_bin = Path("parity")
            package = Path("model.jfm")
            backend = "cuda"
            threads = 6
            max_logit_nrmse = 0.02
            min_logit_cosine = 0.999

        result = {
            "argmax_match": True,
            "normalized_rmse": 0.01,
            "cosine_similarity": 0.998,
        }
        original = native_engine_matrix.run_json
        self.addCleanup(setattr, native_engine_matrix, "run_json", original)
        native_engine_matrix.run_json = lambda _command: result
        with self.assertRaisesRegex(RuntimeError, "logit cosine"):
            native_engine_matrix.run_attention_parity(Args(), [1, 2])

    def test_compare_native_attention_reports_prefill_and_memory_change(self):
        flash = self.metrics(ttft=8.0, itl=9.0, decode=111.0, end_to_end=90.0)
        unfused = self.metrics(ttft=10.0, itl=10.0, decode=100.0, end_to_end=80.0)
        flash.update(prefill_compute_p50_ms=7.0, prefill_scratch_bytes=60)
        unfused.update(prefill_compute_p50_ms=9.0, prefill_scratch_bytes=100)
        comparison = native_engine_matrix.compare_native_attention(flash, unfused)
        self.assertAlmostEqual(comparison["ttft_change_percent"], -20.0)
        self.assertAlmostEqual(comparison["prefill_scratch_change_percent"], -40.0)

    @staticmethod
    def metrics(ttft, itl, decode, end_to_end):
        return {
            "ttft_p50_ms": ttft,
            "itl_p50_ms": itl,
            "decode_tokens_per_second_p50": decode,
            "end_to_end_tokens_per_second_p50": end_to_end,
        }

    @staticmethod
    def raw_run(samples, verified):
        run = {
            field: list(samples) for field in native_engine_matrix.SAMPLE_FIELDS
        }
        run.update(
            prefill_plan_reuse_count=len(samples),
            prefill_attention_backend_verified_count=verified,
            iterations=len(samples),
            warmups=1,
        )
        return run


if __name__ == "__main__":
    unittest.main()
