import csv
import json
import subprocess
import sys
import tempfile
import unittest
import os
import shutil
from pathlib import Path

from experiments.prepare_audit_adjudication import adjudication_work_present


SYSTEMS = ("base", "answer-sft", "no-gate", "full-evisuff")
GROUPS = ("answerable", "uncertainty", "conflict", "insufficient-evidence")
OUTPUT_LABEL_FIELDS = (
    "source_policy_compliant",
    "conflict_disclosure_correct",
    "uncertainty_present",
    "uncertainty_appropriate",
    "stop_decision",
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class HumanAuditPipelineTest(unittest.TestCase):
    def test_started_adjudication_sheet_is_protected(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-adjudication-protection-") as raw:
            path = Path(raw) / "adjudicated_claims.csv"
            write_csv(
                path,
                [{"blind_pair_id": "p1", "support_label": "supported", "rationale": ""}],
                ("blind_pair_id", "support_label", "rationale"),
            )
            self.assertTrue(adjudication_work_present(path, ("support_label",)))

    def test_complete_synthetic_bundle_reaches_all_report_formats(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-summary-test-") as raw:
            root = Path(raw)
            output_master = []
            claim_master = []
            claim_a = []
            claim_b = []
            claim_final = []
            output_a = []
            output_b = []
            output_final = []

            for task_index, group in enumerate(GROUPS):
                task_id = f"task-{task_index}"
                for system_index, system in enumerate(SYSTEMS):
                    output_id = f"{system}::{task_id}"
                    blind_output_id = f"blind-{system_index}-{task_index}"
                    pair_id = f"pair-{system_index}-{task_index}"
                    output_master.append({
                        "output_id": output_id,
                        "blind_output_id": blind_output_id,
                        "system": system,
                        "sample_id": task_id,
                        "group": group,
                        "automatic_E": (system_index + task_index + 1) / 10,
                    })
                    claim_master.append({
                        "blind_pair_id": pair_id,
                        "output_id": output_id,
                        "sample_id": task_id,
                    })
                    final_claim_label = (
                        "supported" if system == "full-evisuff"
                        else "partially_supported" if system == "no-gate"
                        else "unsupported"
                    )
                    claim_final.append({
                        "blind_pair_id": pair_id,
                        "support_label": final_claim_label,
                        "rationale": "adjudicated",
                    })
                    claim_a.append({
                        "blind_pair_id": pair_id,
                        "support_label": final_claim_label,
                        "rationale": "annotator A claim rationale",
                    })
                    claim_b.append({
                        "blind_pair_id": pair_id,
                        "support_label": (
                            "partially_supported"
                            if pair_id == "pair-0-0" else final_claim_label
                        ),
                        "rationale": "annotator B claim rationale",
                    })

                    is_full = system == "full-evisuff"
                    labels = {
                        "blind_output_id": blind_output_id,
                        "source_policy_compliant": "yes" if is_full else "no",
                        "conflict_disclosure_correct": (
                            ("yes" if is_full else "no")
                            if group == "conflict" else "not_applicable"
                        ),
                        "uncertainty_present": (
                            "no" if group == "answerable" else "yes"
                        ),
                        "uncertainty_appropriate": "yes" if is_full else "no",
                        "stop_decision": "appropriate" if is_full else "premature",
                    }
                    output_a.append(dict(labels))
                    output_b.append(dict(labels))
                    output_final.append(dict(labels))
                    output_a[-1]["rationale"] = "annotator A output rationale"
                    output_b[-1]["rationale"] = "annotator B output rationale"
                    output_final[-1]["rationale"] = "adjudicated"

            write_jsonl(root / "outputs.jsonl", output_master)
            write_jsonl(root / "claims.jsonl", claim_master)
            claim_fields = ("blind_pair_id", "support_label", "rationale")
            output_fields = ("blind_output_id",) + OUTPUT_LABEL_FIELDS + ("rationale",)
            for name, rows in (
                ("claim_a.csv", claim_a),
                ("claim_b.csv", claim_b),
                ("claim_final.csv", claim_final),
            ):
                write_csv(root / name, rows, claim_fields)
            for name, rows in (
                ("output_a.csv", output_a),
                ("output_b.csv", output_b),
                ("output_final.csv", output_final),
            ):
                write_csv(root / name, rows, output_fields)

            project_root = Path(__file__).resolve().parents[2]
            adjudication_command = [
                sys.executable,
                str(project_root / "experiments/prepare_audit_adjudication.py"),
                "--annotator-a-claims", str(root / "claim_a.csv"),
                "--annotator-b-claims", str(root / "claim_b.csv"),
                "--annotator-a-outputs", str(root / "output_a.csv"),
                "--annotator-b-outputs", str(root / "output_b.csv"),
                "--output-claims", str(root / "prepared_claim_adjudication.csv"),
                "--output-outputs", str(root / "prepared_output_adjudication.csv"),
                "--output-report", str(root / "adjudication_report.json"),
            ]
            subprocess.run(
                adjudication_command, check=True, capture_output=True, text=True
            )
            with (root / "prepared_claim_adjudication.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                prepared_claims = {
                    row["blind_pair_id"]: row for row in csv.DictReader(handle)
                }
            disputed = prepared_claims["pair-0-0"]
            self.assertEqual(disputed["support_label"], "")
            self.assertEqual(
                disputed["annotator_a_rationale"], "annotator A claim rationale"
            )
            self.assertEqual(
                disputed["annotator_b_rationale"], "annotator B claim rationale"
            )

            command = [
                sys.executable,
                str(project_root / "experiments/summarize_claim_support_audit.py"),
                "--master-claims", str(root / "claims.jsonl"),
                "--master-outputs", str(root / "outputs.jsonl"),
                "--annotator-a-claims", str(root / "claim_a.csv"),
                "--annotator-b-claims", str(root / "claim_b.csv"),
                "--annotator-a-outputs", str(root / "output_a.csv"),
                "--annotator-b-outputs", str(root / "output_b.csv"),
                "--adjudicated-claims", str(root / "claim_final.csv"),
                "--adjudicated-outputs", str(root / "output_final.csv"),
                "--iterations", "100",
                "--seed", "19",
                "--output-json", str(root / "report.json"),
                "--output-md", str(root / "report.md"),
                "--output-tex", str(root / "direct_table.tex"),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete_two_annotator_adjudicated_audit")
            self.assertEqual(report["outputs"], 16)
            self.assertEqual(report["claim_source_pairs"], 16)
            self.assertEqual(set(report["direct"]), set(SYSTEMS))
            self.assertEqual(
                set(report["paired_direct_contrasts"]),
                {"base", "answer-sft", "no-gate"},
            )
            self.assertAlmostEqual(
                report["paired_direct_contrasts"]["base"]["overall"]["metrics"]
                ["claim_support_rate"]["estimate"],
                1.0,
            )
            self.assertIn("Paired Full EviSuff contrasts", (root / "report.md").read_text())
            latex_table = (root / "direct_table.tex").read_text()
            self.assertIn("Direct evidence-sufficiency audit", latex_table)
            self.assertIn("unclear counts as non-support", latex_table)
            self.assertEqual(latex_table.count("Answerable &"), 4)
            self.assertEqual(latex_table.count("Uncertainty &"), 4)
            self.assertEqual(latex_table.count("Conflict &"), 4)
            self.assertEqual(latex_table.count("Insufficient &"), 4)
            self.assertIn("CSR$\\uparrow$", latex_table)
            self.assertIn("Prem.$\\downarrow$", latex_table)

            pdflatex = shutil.which("pdflatex")
            if pdflatex:
                wrapper = root / "table_preflight.tex"
                wrapper.write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{spconf,booktabs}\n"
                    "\\begin{document}\\ninept\\twocolumn\n"
                    "\\input{direct_table.tex}\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )
                env = dict(os.environ)
                template_dir = project_root / "ICASSP2026_Paper_Templates"
                env["TEXINPUTS"] = f"{template_dir}:" + env.get("TEXINPUTS", "")
                subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                log = (root / "table_preflight.log").read_text(encoding="utf-8")
                self.assertNotIn("Overfull", log)
                self.assertNotIn("Float too large", log)


if __name__ == "__main__":
    unittest.main()
