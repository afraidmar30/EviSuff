import unittest

from src.llm.client import _context_window_retry_tokens


class ContextBoundaryTest(unittest.TestCase):
    def test_parses_safe_output_cap_before_quota_classification(self):
        error = RuntimeError(
            "This model's maximum context length is 40960 tokens. "
            "However, you requested 8192 output tokens and your prompt "
            "contains at least 32769 input tokens, for a total of at least 40961 tokens."
        )
        self.assertEqual(_context_window_retry_tokens(error, 8192), 8191)

    def test_ignores_unrelated_quota_error(self):
        self.assertIsNone(
            _context_window_retry_tokens(RuntimeError("429 quota exceeded"), 8192)
        )


if __name__ == "__main__":
    unittest.main()
