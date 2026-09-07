import json
import tempfile
import unittest
from pathlib import Path

from experiments.verify_training_checkpoint_pairing import MATCH_FIELDS, verify


class TrainingCheckpointPairingTest(unittest.TestCase):
    def make_checkpoint(self, root, name, split, *, seed=42):
        checkpoint = root / name
        checkpoint.mkdir()
        args = {field: f"same-{field}" for field in MATCH_FIELDS}
        args.update({
            "model": "/models/answer",
            "seed": seed,
            "data_seed": seed,
            "dataset": [f"/data/{split}/train.jsonl"],
            "val_dataset": [f"/data/{split}/val.jsonl"],
        })
        (checkpoint / "args.json").write_text(json.dumps(args), encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 2058, "max_steps": 2058}), encoding="utf-8"
        )
        return checkpoint

    def test_matched_checkpoints_pass_and_seed_drift_fails(self):
        with tempfile.TemporaryDirectory(prefix="evisuff-training-pair-") as raw:
            root = Path(raw)
            with_gate = self.make_checkpoint(root, "with", "with_gate")
            no_gate = self.make_checkpoint(root, "no", "no_gate")
            report = verify(with_gate, no_gate, "/models/answer", 2058, 42)
            self.assertEqual(report["status"], "passed")

            no_args = json.loads((no_gate / "args.json").read_text(encoding="utf-8"))
            no_args["seed"] = 7
            (no_gate / "args.json").write_text(json.dumps(no_args), encoding="utf-8")
            report = verify(with_gate, no_gate, "/models/answer", 2058, 42)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any(row["field"] == "seed" for row in report["mismatches"]))


if __name__ == "__main__":
    unittest.main()
