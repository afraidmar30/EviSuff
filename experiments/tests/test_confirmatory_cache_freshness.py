import tempfile
import unittest
from pathlib import Path

from experiments.verify_confirmatory_cache_freshness import verify


class ConfirmatoryCacheFreshnessTest(unittest.TestCase):
    def test_stale_cache_fails_but_same_run_resume_passes(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-cache-freshness-") as raw:
            root = Path(raw)
            run_root = root / "runs"
            cache = root / "cache"
            run_root.mkdir()
            cache.mkdir()
            (cache / "web_fetch.md").write_text("cached", encoding="utf-8")
            report = verify(run_root, [f"base={cache}"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["runs"][0]["state"], "stale_cache_without_run")

            trajectories = run_root / "base" / "trajectories"
            trajectories.mkdir(parents=True)
            (trajectories / "task.trajectory.jsonl").write_text(
                '{"event_type":"sample_start"}\n', encoding="utf-8"
            )
            report = verify(run_root, [f"base={cache}"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["runs"][0]["state"], "resumable")

    def test_absent_cache_is_fresh(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-cache-absent-") as raw:
            root = Path(raw)
            report = verify(root / "runs", [f"base={root / 'missing-cache'}"])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["runs"][0]["state"], "fresh")


if __name__ == "__main__":
    unittest.main()
