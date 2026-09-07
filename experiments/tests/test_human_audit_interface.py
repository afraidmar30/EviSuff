import csv
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.render_human_audit_interface import build_html, load_sheet


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class HumanAuditInterfaceTest(unittest.TestCase):
    def test_interface_embeds_blinded_rows_and_escapes_script_close(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-interface-") as raw:
            root = Path(raw)
            claims_path = root / "annotator_a_claim_labels.csv"
            outputs_path = root / "annotator_a_output_labels.csv"
            write_csv(
                claims_path,
                ("blind_pair_id", "question", "claim", "citation_url", "support_label", "rationale"),
                [{"blind_pair_id": "p1", "question": "Q", "claim": "</script>", "citation_url": "https://x.test", "support_label": "", "rationale": ""}],
            )
            write_csv(
                outputs_path,
                ("blind_output_id", "group", "question", "final_answer", "source_policy_compliant", "conflict_disclosure_correct", "uncertainty_present", "uncertainty_appropriate", "stop_decision", "rationale"),
                [{"blind_output_id": "o1", "group": "answerable", "question": "Q", "final_answer": "A", "source_policy_compliant": "", "conflict_disclosure_correct": "", "uncertainty_present": "", "uncertainty_appropriate": "", "stop_decision": "", "rationale": ""}],
            )
            claims = load_sheet(claims_path, "blind_pair_id")
            outputs = load_sheet(outputs_path, "blind_output_id")
            html = build_html(claims, outputs, "instructions", "a")
            payload = re.search(
                r'<script id="bundle" type="application/json">(.*?)</script>', html
            ).group(1)
            bundle = json.loads(payload)
            self.assertEqual(bundle["claims"]["rows"][0]["claim"], "</script>")
            self.assertNotIn("</script>", payload)
            self.assertIn("partially_supported", html)
            self.assertIn("Export current CSV", html)
            changed_claims = json.loads(json.dumps(claims))
            changed_claims["rows"][0]["claim"] = "Different claim under same ID"
            changed_html = build_html(changed_claims, outputs, "instructions", "a")
            changed_payload = re.search(
                r'<script id="bundle" type="application/json">(.*?)</script>',
                changed_html,
            ).group(1)
            self.assertNotEqual(
                bundle["digest"], json.loads(changed_payload)["digest"]
            )

            interface_path = root / "interface.html"
            interface_path.write_text(html, encoding="utf-8")
            scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
            javascript = next(script for script in scripts if "function exportCurrent" in script)
            js_path = root / "interface.js"
            js_path.write_text(javascript, encoding="utf-8")
            node = shutil.which("node")
            if node:
                subprocess.run(
                    [node, "--check", str(js_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            chrome = shutil.which("google-chrome") or shutil.which("chromium")
            if chrome:
                rendered = subprocess.run(
                    [
                        chrome,
                        "--headless",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--dump-dom",
                        interface_path.as_uri(),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
                self.assertIn("<h2>Claim pair 1</h2>", rendered)
                self.assertIn("<option value=\"supported\">supported</option>", rendered)

    def test_interface_refuses_identity_columns(self):
        claims = {"fields": ["blind_pair_id", "system"], "rows": [{"blind_pair_id": "p1", "system": "base"}], "key": "blind_pair_id", "filename": "claims.csv"}
        outputs = {"fields": ["blind_output_id"], "rows": [{"blind_output_id": "o1"}], "key": "blind_output_id", "filename": "outputs.csv"}
        with self.assertRaisesRegex(ValueError, "System-blinding violation"):
            build_html(claims, outputs, "instructions", "a")


if __name__ == "__main__":
    unittest.main()
