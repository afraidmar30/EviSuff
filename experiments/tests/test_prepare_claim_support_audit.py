import json
import csv
import tempfile
import unittest
from pathlib import Path

from experiments.prepare_claim_support_audit import (
    compact_citation_inventory,
    compact_action_trace,
    extract_claim_pairs,
    existing_human_work,
    visible_answer,
)


class VisibleAnswerAuditTest(unittest.TestCase):
    def test_reasoning_urls_are_not_final_citations_or_claim_pairs(self):
        row = {
            "final_answer": (
                "<think>Hidden reasoning cites https://hidden.example/x</think>\n"
                "- The public claim is documented "
                "[by the primary source](https://visible.example/doc)."
            ),
            "citations": [
                {"url": "https://hidden.example/x", "title": "Hidden"},
                {"url": "https://visible.example/doc", "title": "Visible"},
            ],
        }

        answer = visible_answer(row["final_answer"])
        self.assertNotIn("Hidden reasoning", answer)
        self.assertIn("The public claim", answer)

        inventory = {
            item["url"]: item["appears_in_final_answer"]
            for item in compact_citation_inventory(row)
        }
        self.assertFalse(inventory["https://hidden.example/x"])
        self.assertTrue(inventory["https://visible.example/doc"])

        pairs = extract_claim_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["citation_url"], "https://visible.example/doc")

    def test_compact_action_trace_retains_query_and_short_result_context(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-trace-test-") as raw:
            path = Path(raw) / "trajectory.jsonl"
            events = [
                {
                    "step": 4,
                    "event_type": "tool_use",
                    "data": {
                        "tool_use_id": "u1",
                        "tool_name": "web_search",
                        "tool_input": {"query": "primary source query"},
                    },
                },
                {
                    "step": 5,
                    "event_type": "tool_result",
                    "data": {
                        "tool_use_id": "u1",
                        "is_error": False,
                        "result_chars": 500,
                        "result": "Result title\n\nhttps://example.org " + "x" * 500,
                    },
                },
            ]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            trace = compact_action_trace({"trajectory_path": str(path)})
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]["input"]["query"], "primary source query")
            self.assertEqual(trace[0]["result_chars"], 500)
            self.assertLessEqual(len(trace[0]["result_preview"]), 320)

    def test_paragraph_and_following_source_lines_yield_claim_pairs(self):
        row = {
            "final_answer": (
                "The API limit is ten requests per minute.\n"
                "Source: https://docs.example/limit\n\n"
                "A second fact is documented [in the manual]"
                "(https://docs.example/manual). Another sentence is uncited.\n\n"
                "References\n"
                "- https://bibliography.example/unused"
            ),
            "citations": [],
        }
        pairs = extract_claim_pairs(row)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["claim"], "The API limit is ten requests per minute")
        self.assertEqual(pairs[0]["citation_url"], "https://docs.example/limit")
        self.assertIn("A second fact is documented", pairs[1]["claim"])
        self.assertNotIn("Another sentence", pairs[1]["claim"])
        self.assertNotIn("bibliography.example", {pair["citation_url"] for pair in pairs})

    def test_existing_human_labels_are_protected_from_regeneration(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-human-protection-") as raw:
            root = Path(raw)
            path = root / "annotator_a_claim_labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("blind_pair_id", "support_label", "rationale")
                )
                writer.writeheader()
                writer.writerow({
                    "blind_pair_id": "pair-1",
                    "support_label": "supported",
                    "rationale": "opened source",
                })
            self.assertEqual(existing_human_work(root), [path.name])

    def test_blank_templates_can_be_regenerated(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-blank-protection-") as raw:
            root = Path(raw)
            path = root / "annotator_a_claim_labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("blind_pair_id", "support_label", "rationale")
                )
                writer.writeheader()
                writer.writerow({"blind_pair_id": "pair-1", "support_label": "", "rationale": ""})
            self.assertEqual(existing_human_work(root), [])


if __name__ == "__main__":
    unittest.main()
