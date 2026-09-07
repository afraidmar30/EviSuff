import tempfile
import unittest
import zipfile
from pathlib import Path

from experiments.package_human_audit_handoffs import (
    deterministic_zip,
    validate_handoff_bundle,
)


class PackageHumanAuditHandoffsTest(unittest.TestCase):
    @staticmethod
    def bundle(annotator="a"):
        return {
            "annotator": annotator,
            "digest": f"digest-{annotator}",
            "claims": {
                "fields": ["blind_pair_id", "claim", "support_label"],
                "rows": [{"blind_pair_id": "p1", "claim": "Claim"}],
            },
            "outputs": {
                "fields": ["blind_output_id", "final_answer", "stop_decision"],
                "rows": [{"blind_output_id": "o1", "final_answer": "Answer"}],
            },
        }

    def test_bundle_accepts_only_blinded_nonempty_content(self):
        detail = validate_handoff_bundle(self.bundle(), "a")
        self.assertEqual(detail["claims"], 1)
        self.assertEqual(detail["outputs"], 1)

    def test_bundle_rejects_system_or_raw_identity_fields(self):
        bundle = self.bundle()
        bundle["outputs"]["fields"].append("system")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_handoff_bundle(bundle, "a")

    def test_zip_is_deterministic_and_contains_only_declared_files(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-handoff-") as raw:
            root = Path(raw)
            files = {"README.txt": b"instructions", "interface.html": b"<html></html>"}
            left = root / "left.zip"
            right = root / "right.zip"
            deterministic_zip(left, files)
            deterministic_zip(right, files)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            with zipfile.ZipFile(left) as archive:
                self.assertEqual(sorted(archive.namelist()), sorted(files))


if __name__ == "__main__":
    unittest.main()
