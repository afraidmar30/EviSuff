import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class FourSystemConfigPairingTest(unittest.TestCase):
    @staticmethod
    def copy_configs(config_dir, project_root, destination, mutate):
        destination.mkdir()
        (destination / "settings").mkdir()
        for source in config_dir.glob("*.yaml"):
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            settings_source = project_root / payload["settings_path"]
            settings_target = destination / "settings" / settings_source.name
            settings_target.write_text(
                settings_source.read_text(encoding="utf-8"), encoding="utf-8"
            )
            payload["settings_path"] = str(settings_target)
            mutate(source, payload)
            (destination / source.name).write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )

    def test_real_configs_pass_and_controller_drift_fails(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "experiments/verify_four_system_config_pairing.py"
        config_dir = (
            project_root
            / "outputs/experiment_suites/sufficiency_audit25_four_systems"
            / "generated_configs/four_system_sufficiency_audit"
        )
        command = [sys.executable, str(script), "--config-dir", str(config_dir)]
        subprocess.run(command, check=True, capture_output=True, text=True)

        with tempfile.TemporaryDirectory(prefix="evisuff-config-pairing-") as raw:
            copied = Path(raw) / "configs"
            def drift_one_controller(source, payload):
                if source.stem == "audit25_qwen3_no_gate":
                    payload["temperature"] = 0.2
            self.copy_configs(config_dir, project_root, copied, drift_one_controller)
            failed = subprocess.run(
                [sys.executable, str(script), "--config-dir", str(copied)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("non-identity top-level config differs", failed.stdout)

    def test_equal_but_wrong_budget_fails(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "experiments/verify_four_system_config_pairing.py"
        config_dir = (
            project_root
            / "outputs/experiment_suites/sufficiency_audit25_four_systems"
            / "generated_configs/four_system_sufficiency_audit"
        )
        with tempfile.TemporaryDirectory(prefix="evisuff-budget-pairing-") as raw:
            copied = Path(raw) / "configs"
            def wrong_shared_budget(_source, payload):
                payload["max_turns"] = 300
            self.copy_configs(config_dir, project_root, copied, wrong_shared_budget)
            failed = subprocess.run(
                [sys.executable, str(script), "--config-dir", str(copied)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("resolved budget", failed.stdout)


if __name__ == "__main__":
    unittest.main()
