import tempfile
import unittest
from pathlib import Path

from experiments.verify_icassp_submission_ready import (
    collect_local_tex_sources,
    latex_log_issues,
    references_only_on_last_page,
    validate_paper_audit_table,
)


class IcasspSubmissionReadyTest(unittest.TestCase):
    def test_references_must_begin_page_five(self):
        passed, detail = references_only_on_last_page(
            "Conclusion text on page four.\n",
            "6. REFERENCES                    [13] Right-column source\n\n"
            "[1] First source\n[2] Second source\n",
        )
        self.assertTrue(passed, detail)

    def test_local_tex_inputs_are_expanded_at_their_source_position(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-tex-expand-") as raw:
            root = Path(raw)
            (root / "table.tex").write_text(
                r"\label{tab:direct-sufficiency}", encoding="utf-8"
            )
            main = root / "paper.tex"
            main.write_text(
                "before\\input{table}after\\bibliography{refs}", encoding="utf-8"
            )
            expanded = collect_local_tex_sources(main)
            self.assertLess(
                expanded.index(r"\label{tab:direct-sufficiency}"),
                expanded.index(r"\bibliography{refs}"),
            )

    def test_body_text_before_references_or_page_four_references_fails(self):
        passed, _ = references_only_on_last_page(
            "6. REFERENCES\n[1] Early source\n",
            "Stranded conclusion\n6. REFERENCES\n[1] Source\n",
        )
        self.assertFalse(passed)

    def test_latex_log_flags_layout_and_reference_failures(self):
        clean = "Output written on paper.pdf (5 pages)."
        self.assertEqual(latex_log_issues(clean), [])
        issues = latex_log_issues(
            "Overfull \\hbox (2pt too wide)\nLaTeX Warning: There were undefined references."
        )
        self.assertIn("overfull_hbox", issues)
        self.assertIn("undefined_reference", issues)

    def test_compact_audit_table_must_match_report(self):
        metric_names = (
            "claim_support_rate", "source_policy_compliance",
            "uncertainty_calibration_accuracy", "stop_decision_accuracy",
            "premature_finalization_rate",
        )
        systems = ("base", "answer-sft", "no-gate", "full-evisuff")
        report = {
            "direct": {
                system: {"overall": {"metrics": {
                    metric: {"estimate": 0.1 * (index + 1)}
                    for index, metric in enumerate(metric_names)
                }}}
                for system in systems
            },
            "paired_direct_contrasts": {"no-gate": {"overall": {"metrics": {
                metric: {"estimate": 0.1, "ci_low": 0.01, "ci_high": 0.2}
                for metric in metric_names
            }}}},
        }
        text = """\
\\label{tab:human-audit}
System & CSR$\\uparrow$ & SPC$\\uparrow$ & U-Cal$\\uparrow$ & Stop$\\uparrow$ & Prem.$\\downarrow$ \\\\
Base & 10.0 & 20.0 & 30.0 & 40.0 & 50.0 \\\\
Answer-SFT & 10.0 & 20.0 & 30.0 & 40.0 & 50.0 \\\\
No-gate & 10.0 & 20.0 & 30.0 & 40.0 & 50.0 \\\\
Full EviSuff & 10.0 & 20.0 & 30.0 & 40.0 & 50.0 \\\\
Full $-$ No-gate & +10.0 [1.0,20.0] & +10.0 [1.0,20.0] & +10.0 [1.0,20.0] & +10.0 [1.0,20.0] & +10.0 [1.0,20.0] \\\\
"""
        passed, detail = validate_paper_audit_table(text, report)
        self.assertTrue(passed, detail)
        passed, _ = validate_paper_audit_table(text.replace("50.0", "49.0", 1), report)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
