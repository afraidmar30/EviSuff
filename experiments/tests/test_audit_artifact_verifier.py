import unittest
import tempfile
from pathlib import Path

from experiments.verify_sufficiency_audit_artifacts import (
    conflict_label_matches_group,
    load_interface_bundle,
    sheet_content_mismatches,
    validate_core_latex_table,
)


class AuditArtifactVerifierTest(unittest.TestCase):
    @staticmethod
    def _core_table() -> str:
        rows = []
        for system in ("Base", "Answer-SFT", "No-gate", "Full EviSuff"):
            for condition in ("Answerable", "Uncertainty", "Conflict", "Insufficient"):
                cda = "50.0" if condition == "Conflict" else "--"
                rows.append(
                    f"{system} & {condition} & 50.0 & 50.0 & {cda} & "
                    "50.0 & 50.0 & 50.0 & 50.0 \\\\"
                )
        return "\n".join([
            r"\begin{table*}[t]",
            r"\caption{Direct evidence-sufficiency audit (\%; arrows show the favorable direction).}",
            r"\label{tab:direct-sufficiency}",
            r"System & Condition & CSR$\uparrow$ & SPC$\uparrow$ & CDA$\uparrow$ & U-Cal$\uparrow$ & Stop$\uparrow$ & Prem.$\downarrow$ & Unnec.$\downarrow$ \\",
            *rows,
            r"\end{table*}",
        ])

    @staticmethod
    def _core_report() -> dict:
        conditions = ("answerable", "uncertainty", "conflict", "insufficient-evidence")
        metrics = (
            "claim_support_rate", "source_policy_compliance",
            "conflict_disclosure_accuracy", "uncertainty_calibration_accuracy",
            "stop_decision_accuracy", "premature_finalization_rate",
            "unnecessary_search_rate",
        )
        return {"direct": {
            system: {
                condition: {"metrics": {
                    metric: {"estimate": None if metric == "conflict_disclosure_accuracy" and condition != "conflict" else 0.5}
                    for metric in metrics
                }}
                for condition in conditions
            }
            for system in ("base", "answer-sft", "no-gate", "full-evisuff")
        }}

    def test_core_latex_table_requires_16_report_aligned_rows(self):
        passed, detail = validate_core_latex_table(self._core_table(), self._core_report())
        self.assertTrue(passed, detail)

    def test_core_latex_table_rejects_direction_or_value_drift(self):
        broken = self._core_table().replace(r"Prem.$\downarrow$", r"Prem.$\uparrow$")
        broken = broken.replace("Base & Conflict & 50.0", "Base & Conflict & 49.9", 1)
        passed, detail = validate_core_latex_table(broken, self._core_report())
        self.assertFalse(passed)
        self.assertIn("prem_down", detail["missing_tokens"])
        self.assertTrue(detail["report_mismatches"])

    def test_interface_bundle_loader_rejects_missing_payload(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-interface-verify-") as raw:
            path = Path(raw) / "interface.html"
            path.write_text("<html>no bundle</html>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing embedded annotation bundle"):
                load_interface_bundle(path)

    def test_conflict_label_must_match_condition(self):
        self.assertTrue(conflict_label_matches_group("yes", "conflict"))
        self.assertTrue(conflict_label_matches_group("no", "conflict"))
        self.assertFalse(conflict_label_matches_group("not_applicable", "conflict"))
        self.assertTrue(conflict_label_matches_group("not_applicable", "answerable"))
        self.assertFalse(conflict_label_matches_group("no", "uncertainty"))

    def test_sheet_content_comparison_detects_accidental_edits(self):
        master = [{
            "blind_pair_id": "p1",
            "question": "Original question",
            "claim": "Original claim",
            "search_calls": 0,
        }]
        clean = [{
            "blind_pair_id": "p1",
            "question": "Original question",
            "claim": "Original claim",
            "search_calls": "0",
            "support_label": "supported",
        }]
        fields = {"question", "claim", "search_calls"}
        self.assertEqual(
            sheet_content_mismatches(master, clean, "blind_pair_id", fields), []
        )
        edited = [dict(clean[0], claim="Edited after labeling")]
        mismatches = sheet_content_mismatches(
            master, edited, "blind_pair_id", fields
        )
        self.assertEqual(mismatches[0]["field"], "claim")

    def test_sheet_content_comparison_detects_id_drift(self):
        master = [{"blind_pair_id": "p1", "claim": "Claim"}]
        sheet = [{"blind_pair_id": "p2", "claim": "Claim"}]
        mismatches = sheet_content_mismatches(
            master, sheet, "blind_pair_id", {"claim"}
        )
        self.assertTrue(any(row["reason"] == "ID set differs" for row in mismatches))


if __name__ == "__main__":
    unittest.main()
