import unittest

from tools.bench import runtime_jsonl


class RuntimeJSONLTest(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(runtime_jsonl.percentile([4.0, 1.0, 3.0, 2.0], 0.50), 2.0)
        self.assertEqual(runtime_jsonl.percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)

    def test_summary_separates_decode_from_end_to_end_rate(self):
        result = runtime_jsonl.RequestResult(
            latency_ms=1000.0,
            ttft_ms=400.0,
            inter_token_latency_ms=[100.0, 100.0, 100.0],
            sampled_tokens=[1, 2, 3, 4],
            generated_text="test",
            finish_reason="length",
            prompt_tokens=8,
            completion_tokens=4,
            events=[],
        )
        summary = runtime_jsonl.summarize([result], 1.0)
        self.assertEqual(summary["output_token_throughput"], 4.0)
        self.assertEqual(summary["end_to_end_tokens_per_second"], 4.0)
        self.assertEqual(summary["decode_tokens_per_second"], 10.0)

    def test_equivalence_rejects_a_different_sequence(self):
        first = self.result([1, 2])
        second = self.result([1, 3])
        with self.assertRaisesRegex(RuntimeError, "differ from the reference"):
            runtime_jsonl.validate_equivalent_tokens([first, second])

    @staticmethod
    def result(tokens):
        return runtime_jsonl.RequestResult(
            latency_ms=1.0,
            ttft_ms=1.0,
            inter_token_latency_ms=[],
            sampled_tokens=tokens,
            generated_text="",
            finish_reason="length",
            prompt_tokens=1,
            completion_tokens=len(tokens),
            events=[],
        )


if __name__ == "__main__":
    unittest.main()
