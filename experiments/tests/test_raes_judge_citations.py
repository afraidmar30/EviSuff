import unittest

from src.raes_eval.judge import citation_urls


class FinalCitationScoringTest(unittest.TestCase):
    def test_only_visible_final_answer_urls_count_as_citations(self):
        result = {
            "final_answer": (
                "<think>Reasoning used https://reasoning.example/private</think>\n"
                "The claim is supported by "
                "[the documentation](https://Example.org/doc/#section)."
            ),
            "citations": [
                {"url": "https://retrieved.example/not-used", "cited": False},
                {"url": "https://Example.org/doc/#section", "cited": False},
            ],
        }

        self.assertEqual(citation_urls(result), ["https://example.org/doc"])


if __name__ == "__main__":
    unittest.main()
