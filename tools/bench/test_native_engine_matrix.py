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

    def test_validate_pair_rejects_token_mismatch(self):
        native = {
            "source_sha256": "abc",
            "prompt_tokens": [1],
            "sampled_tokens": [2],
        }
        llama = {"prompt_tokens": [1], "sampled_tokens": [3]}
        with self.assertRaisesRegex(RuntimeError, "greedy token IDs differ"):
            native_engine_matrix.validate_pair(native, llama, "abc")

    @staticmethod
    def metrics(ttft, itl, decode, end_to_end):
        return {
            "ttft_p50_ms": ttft,
            "itl_p50_ms": itl,
            "decode_tokens_per_second_p50": decode,
            "end_to_end_tokens_per_second_p50": end_to_end,
        }


if __name__ == "__main__":
    unittest.main()
